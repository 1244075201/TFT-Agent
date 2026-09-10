"""Searchable entity selection models and their optional Qt widget."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from tft_agent.desktop.catalog import CanonicalEntityCatalog, EntityOption


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


def searchable_selector_class() -> Any:
    """Build the Qt widget lazily so model tests never require a GUI runtime."""
    from PySide6.QtCore import QStringListModel, Qt, Signal
    from PySide6.QtWidgets import QCompleter, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

    class SearchableEntitySelector(QWidget):
        values_changed = Signal()

        def __init__(self, model: EntitySelectorModel) -> None:
            super().__init__()
            self.model = model
            self.search = QLineEdit()
            self.search.setPlaceholderText("搜索名称或别名…")
            self.selected = QLabel()
            self.remove_button = QPushButton("移除末项")
            self.remove_button.setVisible(model.multiple)
            self.completion_model = QStringListModel()
            self.completer = QCompleter(self.completion_model, self)
            self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
            self.search.setCompleter(self.completer)
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.selected)
            layout.addWidget(self.search)
            layout.addWidget(self.remove_button)
            self.search.textChanged.connect(self._refresh)
            self.completer.activated.connect(self._activate)
            self.remove_button.clicked.connect(self._remove_last)
            self._refresh("")

        def _refresh(self, query: str) -> None:
            self.completion_model.setStringList([
                option.display_name for option in self.model.suggestions(query)
            ])

        def _activate(self, name: str) -> None:
            option = next(
                value for value in self.model.suggestions(name) if value.display_name == name
            )
            self.model.select(option)
            self.selected.setText("，".join(self.model.display_values()))
            if self.model.multiple:
                self.search.clear()
            else:
                self.search.setText(option.display_name)
            self.values_changed.emit()

        def _remove_last(self) -> None:
            if self.model.selected:
                self.model.remove(self.model.selected[-1].canonical_id)
                self.selected.setText("，".join(self.model.display_values()))
                self.values_changed.emit()

        def text(self) -> str:
            selected = self.model.display_values()
            return "，".join(selected) if selected else self.search.text()

        def setText(self, value: str) -> None:
            self.search.setText(value)

        def clear(self) -> None:
            self.model.clear()
            self.selected.clear()
            self.search.clear()

    return SearchableEntitySelector
