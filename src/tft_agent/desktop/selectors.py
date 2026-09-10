"""Searchable entity selection models and their optional Qt widget."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from tft_agent.desktop.catalog import CanonicalEntityCatalog, EntityOption, search_match_reason


class EntitySelectorModel:
    """GUI-independent filtering and selected-value state."""

    def __init__(
        self,
        catalog: CanonicalEntityCatalog,
        entity_types: Iterable[str],
        *,
        multiple: bool = False,
        predicate: Callable[[EntityOption], bool] | None = None,
    ) -> None:
        self.catalog = catalog
        self.entity_types = frozenset(entity_types)
        self.multiple = multiple
        self.predicate = predicate
        self.selected: list[EntityOption] = []

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
            self.selected = [option]
        elif all(item.canonical_id != option.canonical_id for item in self.selected):
            self.selected.append(option)

    def remove(self, canonical_id: str) -> None:
        self.selected = [item for item in self.selected if item.canonical_id != canonical_id]

    def clear(self) -> None:
        self.selected.clear()

    def display_values(self) -> list[str]:
        return [item.display_name for item in self.selected]


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
    from PySide6.QtCore import QStringListModel, Qt, Signal
    from PySide6.QtWidgets import QCompleter, QHBoxLayout, QLineEdit, QPushButton, QWidget

    class SearchableEntitySelector(QWidget):
        values_changed = Signal()

        def __init__(self, model: EntitySelectorModel) -> None:
            super().__init__()
            self.model = model
            self.search = QLineEdit()
            self.search.setPlaceholderText("搜索名称或别名…")
            self.chips = QWidget()
            self.chips_layout = QHBoxLayout(self.chips)
            self.chips_layout.setContentsMargins(0, 0, 0, 0)
            self.completion_model = QStringListModel()
            self.completion_options: dict[str, EntityOption] = {}
            self.completer = QCompleter(self.completion_model, self)
            self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
            self.search.setCompleter(self.completer)
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.chips)
            layout.addWidget(self.search)
            self.search.textChanged.connect(self._refresh)
            self.completer.activated.connect(self._activate)
            self._refresh("")

        def _refresh(self, query: str) -> None:
            self.completion_options = {
                f"{option.display_name}　·　{search_match_reason(option, query)}": option
                for option in self.model.suggestions(query)
            }
            self.completion_model.setStringList(list(self.completion_options))

        def _activate(self, label: str) -> None:
            option = self.completion_options[label]
            self.model.select(option)
            self._render_chips()
            self.search.clear()
            self.values_changed.emit()

        def _render_chips(self) -> None:
            while self.chips_layout.count():
                item = self.chips_layout.takeAt(0)
                widget = item.widget() if item is not None else None
                if widget is not None:
                    widget.deleteLater()
            for option in self.model.selected:
                chip = QPushButton(f"{option.display_name} ×")
                chip.setToolTip(option.entity_type)
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
            selected = self.model.display_values()
            return "，".join(selected) if selected else self.search.text()

        def setText(self, value: str) -> None:
            self.model.clear()
            unresolved: list[str] = []
            for name in (item.strip() for item in value.replace("，", ",").split(",")):
                if not name:
                    continue
                match = next(
                    (item for item in self.model.suggestions(name) if item.display_name == name),
                    None,
                )
                if match is not None:
                    self.model.select(match)
                else:
                    unresolved.append(name)
            self._render_chips()
            self.search.setText("，".join(unresolved))

        def clear(self) -> None:
            self.model.clear()
            self._render_chips()
            self.search.clear()

    return SearchableEntitySelector
