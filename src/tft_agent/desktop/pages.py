"""Product-mode pages and match-state editor layout."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QLabel, QLineEdit, QTabWidget, QVBoxLayout, QWidget

from tft_agent.app.pipeline import ProductMode
from tft_agent.desktop.catalog import CanonicalEntityCatalog, is_completed_item, is_component
from tft_agent.desktop.selectors import EntitySelectorModel, searchable_selector_class


class ProductPages(QTabWidget):
    mode_changed = Signal(str)

    def __init__(self, catalog: CanonicalEntityCatalog) -> None:
        super().__init__()
        self.state_inputs: dict[str, Any] = {}
        self._modes = [
            ProductMode.GENERAL_GUIDE,
            ProductMode.COMP_BROWSER,
            ProductMode.MATCH_GUIDANCE,
        ]
        self.addTab(self._information_page("提问完整攻略，不读取或修改当前对局状态。"), "通用攻略")
        self.addTab(self._information_page("浏览当前版本阵容；结果来自版本化目录。"), "阵容推荐")
        self.addTab(self._match_page(catalog), "对局指导")
        self.setCurrentIndex(2)
        self.currentChanged.connect(self._emit_mode)

    def _information_page(self, text: str) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel(text))
        layout.addStretch()
        return page

    def _match_page(self, catalog: CanonicalEntityCatalog) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        Selector = searchable_selector_class()
        field_types = {
            "comp_name": ("composition",),
            "augments": ("augment",),
            "augment_choices": ("augment",),
            "item_components": ("item",),
            "completed_items": ("item",),
            "emblems": ("emblem",),
            "core_units": ("champion",),
        }
        predicates = {
            "item_components": is_component,
            "completed_items": is_completed_item,
        }
        labels = (
            ("comp_name", "目标阵容"), ("stage", "阶段"), ("level", "等级"),
            ("gold", "金币"), ("hp", "生命"), ("augments", "已有强化"),
            ("augment_choices", "强化三选一"), ("item_components", "散件"),
            ("completed_items", "成装"), ("emblems", "转职"),
            ("core_units", "关键棋子"), ("notes", "备注"),
        )
        for key, label in labels:
            types = field_types.get(key)
            editor: Any = (
                Selector(EntitySelectorModel(
                    catalog,
                    types,
                    multiple=key != "comp_name",
                    predicate=predicates.get(key),
                ))
                if types
                else QLineEdit()
            )
            self.state_inputs[key] = editor
            form.addRow(label, editor)
        return page

    @property
    def product_mode(self) -> ProductMode:
        return self._modes[self.currentIndex()]

    def _emit_mode(self, index: int) -> None:
        self.mode_changed.emit(self._modes[index].value)
