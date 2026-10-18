"""Versioned DataTFT candidate ingestion with validation and atomic activation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from tft_agent.core.config import Settings
from tft_agent.core.update_state import database_lease
from tft_agent.ingestion.sources.datatft import DataTFTSource, RawHTTPResponse
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)
from tft_agent.tft_mcp_server.schemas import (
    ActivePatchContext,
    CompStatsRecord,
    DataTFTSourceResponse,
)


async def sync(
    input_path: Path | None,
    database: Path,
    settings: Settings,
    requested_patch: str | None = None,
    set_number: int | None = None,
    set_name: str | None = None,
    activate: bool = True,
) -> tuple[str, int]:
    async with database_lease(database):
        return await _sync(input_path, database, settings, requested_patch,
                           set_number, set_name, activate)


async def _sync(
    input_path: Path | None, database: Path, settings: Settings,
    requested_patch: str | None, set_number: int | None, set_name: str | None,
    activate: bool,
) -> tuple[str, int]:
    repository = SQLiteGameDataRepository(database.resolve())
    await repository.initialize()
    current = await repository.active_patch_context()
    if current is None:
        current = await repository.bootstrap_patch_registry(
            settings.current_game_patch, settings.current_set_number, settings.current_set_name
        )
    if current is None:
        current = ActivePatchContext(
            game_patch=settings.current_game_patch,
            patch_family=settings.current_game_patch.rstrip("abcdefghijklmnopqrstuvwxyz"),
            set_number=settings.current_set_number,
            set_name=settings.current_set_name,
            active_snapshot_id="",
        )
    selected_set_number = set_number or current.set_number
    selected_set_name = set_name or current.set_name
    expected_patch = requested_patch
    sync_id = f"datatft-sync:{uuid4()}"
    started_at = datetime.now(UTC)
    await repository.create_sync_run(
        sync_id, expected_patch, settings.datatft_server,
        selected_set_number, selected_set_name, started_at,
    )
    source = DataTFTSource(str(settings.datatft_url), settings.datatft_server)
    raw_responses: list[RawHTTPResponse] = []
    stored_responses: list[DataTFTSourceResponse] = []
    detected_patch: str | None = None
    try:
        if input_path is None:
            payload, etag = await asyncio.to_thread(
                source.fetch_current_with_raw,
                raw_responses,
                expected_patch,
                settings.datatft_request_interval_seconds,
            )
            source_url = str(settings.datatft_url)
            raw_status = "AVAILABLE"
        else:
            payload = input_path.read_text(encoding="utf-8")
            etag = None
            source_url = input_path.resolve().as_uri()
            raw_status = "CANONICAL_INPUT_ONLY"
            raw_responses.append(RawHTTPResponse(
                endpoint_type="OFFLINE_CANONICAL_INPUT", request_url=source_url,
                request_key=None, retrieved_at=datetime.now(UTC), http_status=200,
                etag=None, content_type="application/json", body=payload,
            ))
        metadata, records = source.parse(payload, source_url, etag)
        detected_patch = metadata.game_patch
        if requested_patch is not None and detected_patch != requested_patch:
            raise ValueError(
                f"requested patch {requested_patch!r} does not match payload {detected_patch!r}"
            )
        stored_responses = _source_records(sync_id, detected_patch, raw_responses)
        await repository.store_source_responses(stored_responses)
        await repository.update_sync_run(sync_id, "fetched", detected_patch=detected_patch)
        await repository.register_candidate_patch(
            detected_patch, selected_set_number, selected_set_name, "datatft", started_at
        )
        response_ids = {
            item.request_key: item.response_id for item in stored_responses
            if item.endpoint_type == "COMPOSITION_DETAIL" and item.request_key is not None
        }
        snapshot_id, _created = await repository.store_candidate_snapshot(
            metadata, records, payload, sync_id, response_ids, raw_status
        )
        warnings = await _validate(repository, current.active_snapshot_id, records)
        if not activate:
            await repository.update_sync_run(
                sync_id, "validated", detected_patch=detected_patch,
                snapshot_id=snapshot_id, warnings=[*warnings, "server_version_unverified"],
            )
            return snapshot_id, len(records)
        if not _created and current.active_snapshot_id == snapshot_id:
            await repository.update_sync_run(
                sync_id, "activated", detected_patch=detected_patch,
                snapshot_id=snapshot_id, warnings=["unchanged_content"],
            )
            return snapshot_id, len(records)
        await repository.update_sync_run(
            sync_id, "validated", detected_patch=detected_patch,
            snapshot_id=snapshot_id, warnings=warnings,
        )
        await repository.publish_snapshot(
            detected_patch, selected_set_number, selected_set_name,
            snapshot_id, sync_id, warnings,
        )
        return snapshot_id, len(records)
    except Exception as exc:
        if raw_responses and not stored_responses:
            stored_responses = _source_records(
                sync_id, detected_patch or expected_patch, raw_responses
            )
            await repository.store_source_responses(stored_responses)
        await repository.update_sync_run(
            sync_id, "failed", detected_patch=detected_patch,
            failure_stage="ingestion", failure_code=type(exc).__name__,
            failure_message=str(exc),
        )
        raise


def _source_records(
    sync_id: str, game_patch: str | None, responses: list[RawHTTPResponse]
) -> list[DataTFTSourceResponse]:
    return [DataTFTSourceResponse(
        response_id=f"{sync_id}:response:{index}", sync_id=sync_id,
        game_patch=game_patch, endpoint_type=item.endpoint_type,
        request_url=item.request_url, request_key=item.request_key,
        retrieved_at=item.retrieved_at, http_status=item.http_status, etag=item.etag,
        content_hash=hashlib.sha256(item.body.encode("utf-8")).hexdigest(),
        content_type=item.content_type, response_body=item.body,
        parse_status="parsed" if 200 <= item.http_status < 300 else "http_error",
        error_message=None if 200 <= item.http_status < 300 else f"HTTP {item.http_status}",
    ) for index, item in enumerate(responses)]


async def _validate(
    repository: SQLiteGameDataRepository,
    previous_snapshot_id: str,
    records: list[CompStatsRecord],
) -> list[str]:
    identifiers = [item.comp_id for item in records]
    names = [item.name for item in records]
    if not identifiers or any(not value.strip() for value in identifiers + names):
        raise ValueError("candidate compositions require non-empty IDs and names")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate composition IDs must be unique")
    warnings: list[str] = []
    previous_count = await repository.snapshot_comp_count(previous_snapshot_id)
    if previous_count and len(records) < previous_count:
        ratio = len(records) / previous_count
        if ratio < 0.5 and previous_count - len(records) >= 3:
            raise ValueError(
                "candidate composition count is catastrophically below active snapshot"
            )
        if ratio < 0.8:
            warnings.append("composition count is below 80% of the active snapshot")
    if all(not item.augments for item in records):
        warnings.append("all augment recommendations are empty")
    if all(item.tier is None for item in records):
        warnings.append("all composition tiers are empty")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize and activate a DataTFT patch")
    parser.add_argument("--input", type=Path, help="Reviewed canonical JSON snapshot")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--patch", help="Expected exact TFT patch, for example 18.2b")
    parser.add_argument("--set-number", type=int)
    parser.add_argument("--set-name")
    args = parser.parse_args()
    settings = Settings()
    snapshot_id, count = asyncio.run(sync(
        args.input, args.database or settings.database_path, settings,
        args.patch, args.set_number, args.set_name,
    ))
    print(f"snapshot_id={snapshot_id} comps={count}")


if __name__ == "__main__":
    main()
