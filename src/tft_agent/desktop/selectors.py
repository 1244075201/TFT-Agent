"""Searchable entity selection models and their optional Qt widget."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from pydantic import Field

from tft_agent.core.schemas import StrictModel
from tft_agent.desktop.catalog import CanonicalEntityCatalog, EntityOption, search_match_reason


class SelectedEntity(StrictModel):
    canonical_id: str
    display_name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    quantity: int = Field(default=1, ge=1)


class EntitySelectorModel:
    """GUI-independent filtering and selected-value state."""

    def __init__(
        self,
        catalog: CanonicalEntityCatalog,
        entity_types: Iterable[str],
        *,
        multiple: bool = False,
        count_duplicates: bool = False,
        search_placeholder: str = "搜索名称或别名…",
        selection_label: str = "已添加：",
        predicate: Callable[[EntityOption], bool] | None = None,
    ) -> None:
        self.catalog = catalog
        self.entity_types = frozenset(entity_types)
        self.multiple = multiple
        self.count_duplicates = count_duplicates
        self.predicate = predicate
        self.search_placeholder = search_placeholder
        self.selection_label = selection_label
        self.selected: list[SelectedEntity] = []

    def suggestions(self, query: str, limit: int = 30) -> list[EntityOption]:
        options = [
            option
            for option in self.catalog.options
            if option.entity_type in self.entity_types
            and (self.predicate is None or self.predicate(option))
        ]
        return self.catalog.search(query, options, limit)

    def select(self, option: EntityOption) -> None:
        if option.entity_type not in self.entity_types or (
            self.predicate is not None and not self.predicate(option)
        ):
            raise ValueError(f"unsupported entity type: {option.entity_type}")
        if not self.multiple:
            self.selected = [SelectedEntity(
                canonical_id=option.canonical_id, display_name=option.display_name,
                entity_type=option.entity_type, aliases=option.aliases,
            )]
        else:
            existing = next((item for item in self.selected
                             if item.canonical_id == option.canonical_id), None)
            if existing is None:
                self.selected.append(SelectedEntity(
                    canonical_id=option.canonical_id, display_name=option.display_name,
                    entity_type=option.entity_type, aliases=option.aliases,
                ))
            elif self.count_duplicates:
                existing.quantity += 1

    def remove(self, canonical_id: str) -> None:
        selected = next((item for item in self.selected if item.canonical_id == canonical_id), None)
        if selected is not None and selected.quantity > 1:
            selected.quantity -= 1
        else:
            self.selected = [item for item in self.selected if item.canonical_id != canonical_id]

    def clear(self) -> None:
        self.selected.clear()

    def display_values(self) -> list[str]:
        return [item.display_name for item in self.selected for _ in range(item.quantity)]

    def set_display_values(self, values: Iterable[str]) -> None:
        self.clear()
        for name in values:
            match = next(
                (item for item in self.suggestions(name) if item.display_name == name), None
            )
            if match is not None:
                self.select(match)
                continue
            entity_type = next(iter(self.entity_types), "unknown")
            existing = next((item for item in self.selected if item.display_name == name), None)
            if existing is not None and self.count_duplicates:
                existing.quantity += 1
            elif existing is None:
                self.selected.append(SelectedEntity(
                    canonical_id=f"legacy:{entity_type}:{name}", display_name=name,
                    entity_type=entity_type,
                ))


class AugmentChoiceModel:
    """Stable three-slot augment selection with cross-slot deduplication."""

    def __init__(self) -> None:
        self.slots: list[EntityOption | None] = [None, None, None]

    def choose(self, index: int, option: EntityOption) -> None:
        if option.entity_type != "augment":
            raise ValueError("augment choices require augment entities")
        if any(
            value is not None and value.canonical_id == option.canonical_id
            for position, value in enumerate(self.slots)
            if position != index
        ):
            raise ValueError("the same augment cannot occupy two choices")
        self.slots[index] = option

    def clear_slot(self, index: int) -> None:
        self.slots[index] = None

    def display_values(self) -> list[str]:
        return [value.display_name for value in self.slots if value is not None]


def searchable_selector_class() -> Any:
    """Build the Qt widget lazily so model tests never require a GUI runtime."""
    from PySide6.QtCore import QModelIndex, Signal
    from PySide6.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    class SearchableEntitySelector(QWidget):
        values_changed = Signal()

        def __init__(self, model: EntitySelectorModel) -> None:
            super().__init__()
            self.model = model
            self.search = QLineEdit()
            self.search.setPlaceholderText(model.search_placeholder)
            self.candidates = QListWidget()
            self.candidates.setMaximumHeight(112)
            self.candidates.hide()
            self.chips = QWidget()
            self.chips_layout = QHBoxLayout(self.chips)
            self.chips_layout.setContentsMargins(0, 0, 0, 0)
            self.selected_label = QLabel(model.selection_label)
            selected_row = QHBoxLayout()
            selected_row.setContentsMargins(0, 0, 0, 0)
            selected_row.addWidget(self.selected_label)
            selected_row.addWidget(self.chips, 1)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.search)
            layout.addWidget(self.candidates)
            layout.addLayout(selected_row)
            self.search.textEdited.connect(self._refresh)
            self.candidates.clicked.connect(self._activate)
            self.candidates.activated.connect(self._activate)
            self._candidate_options: list[EntityOption] = []
            self._render_chips()

        def _refresh(self, query: str) -> None:
            self.candidates.clear()
            self._candidate_options = self.model.suggestions(query) if query.strip() else []
            for option in self._candidate_options:
                self.candidates.addItem(
                    f"{option.display_name}　·　{search_match_reason(option, query)}"
                )
            self.candidates.setVisible(bool(self._candidate_options))

        def _activate(self, index: QModelIndex) -> None:
            option = self._candidate_options[index.row()]
            self.model.select(option)
            self._render_chips()
            self.search.clear()
            self.candidates.clear()
            self.candidates.hide()
            self._candidate_options.clear()
            self.values_changed.emit()

        def _render_chips(self) -> None:
            while self.chips_layout.count():
                item = self.chips_layout.takeAt(0)
                widget = item.widget() if item is not None else None
                if widget is not None:
                    widget.deleteLater()
            for option in self.model.selected:
                quantity = f" ×{option.quantity}" if option.quantity > 1 else ""
                chip = QPushButton(f"{option.display_name}{quantity} ×")
                chip.clicked.connect(
                    lambda _checked=False, identifier=option.canonical_id:
                    self._remove(identifier)
                )
                self.chips_layout.addWidget(chip)

        def _remove(self, canonical_id: str) -> None:
            self.model.remove(canonical_id)
            self._render_chips()
            if not self.model.multiple:
                self.search.clear()
            self.values_changed.emit()

        def text(self) -> str:
            return "，".join(self.model.display_values())

        def setText(self, value: str) -> None:
            values = [item.strip() for item in value.replace("，", ",").split(",")
                      if item.strip()]
            self.model.set_display_values(values)
            self._render_chips()
            self.search.clear()
            self.candidates.hide()

        def clear(self) -> None:
            self.model.clear()
            self._render_chips()
            self.search.clear()

    return SearchableEntitySelector
