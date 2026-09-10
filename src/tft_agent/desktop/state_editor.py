"""Typed, scrollable editor for user-confirmed match facts."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tft_agent.core.schemas import GameState
from tft_agent.desktop.catalog import (
    CanonicalEntityCatalog,
    is_artifact,
    is_completed_item,
    is_component,
    is_radiant_item,
)
from tft_agent.desktop.selectors import EntitySelectorModel, searchable_selector_class


class NumericStateEditor(QSpinBox):
    def __init__(self, maximum: int) -> None:
        super().__init__()
        self.setRange(-1, maximum)
        self.setSpecialValueText("未填写")

    def optional_value(self) -> int | None:
        return None if self.value() < 0 else self.value()


class StageSelector(QComboBox):
    def __init__(self) -> None:
        super().__init__()
        self.addItem("未填写", None)
        for stage in range(2, 8):
            for round_number in range(1, 8):
                value = f"{stage}-{round_number}"
                self.addItem(value, value)

    def optional_value(self) -> str | None:
        value = self.currentData()
        return str(value) if value else None


class GameStateEditor(QScrollArea):
    dirty_changed = Signal(bool)
    save_requested = Signal()
    clear_requested = Signal()
    screenshot_requested = Signal()

    def __init__(self, catalog: CanonicalEntityCatalog) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self._dirty = False
        self._augment_choices: list[str] = []
        body = QWidget()
        self.setWidget(body)
        layout = QVBoxLayout(body)
        layout.setSpacing(6)
        form = QFormLayout()
        form.setVerticalSpacing(6)
        layout.addLayout(form)
        Selector = searchable_selector_class()
        self.comp = Selector(EntitySelectorModel(
            catalog, ["composition"], search_placeholder="搜索阵容名称或别名…",
            selection_label="当前选择：",
        ))
        self.stage = StageSelector()
        self.level = NumericStateEditor(10)
        self.gold = NumericStateEditor(200)
        self.hp = NumericStateEditor(100)
        self.augments = Selector(EntitySelectorModel(
            catalog, ["augment"], multiple=True, search_placeholder="搜索强化名称或别名…"
        ))
        self.components = Selector(EntitySelectorModel(
            catalog, ["item"], multiple=True, count_duplicates=True, predicate=is_component,
            search_placeholder="搜索散件名称或别名…",
        ))
        self.completed = Selector(EntitySelectorModel(
            catalog, ["item"], multiple=True, count_duplicates=True,
            predicate=is_completed_item, search_placeholder="搜索成装名称或别名…",
        ))
        self.artifacts = Selector(EntitySelectorModel(
            catalog, ["item"], multiple=True, count_duplicates=True,
            predicate=is_artifact, search_placeholder="搜索奥恩神器名称或别名…",
        ))
        self.radiant_items = Selector(EntitySelectorModel(
            catalog, ["item"], multiple=True, count_duplicates=True,
            predicate=is_radiant_item, search_placeholder="搜索光明装备名称或别名…",
        ))
        self.emblems = Selector(EntitySelectorModel(
            catalog, ["emblem"], multiple=True, count_duplicates=True,
            search_placeholder="搜索转职名称或别名…",
        ))
        self.units = Selector(EntitySelectorModel(
            catalog, ["champion"], multiple=True, search_placeholder="搜索棋子名称或别名…"
        ))
        self.notes = QLineEdit()
        for label, widget in (
            ("目标阵容", self.comp), ("阶段", self.stage), ("等级", self.level),
            ("金币", self.gold), ("生命", self.hp), ("已有强化", self.augments),
            ("散件", self.components), ("成装", self.completed),
            ("奥恩神器", self.artifacts), ("光明装备", self.radiant_items),
            ("转职", self.emblems),
            ("关键棋子", self.units), ("备注", self.notes),
        ):
            form.addRow(label, widget)
        self.conflict = QLabel()
        self.conflict.setStyleSheet("color:#b7791f")
        layout.addWidget(self.conflict)
        actions = QHBoxLayout()
        save = QPushButton("保存局面")
        clear = QPushButton("清空本局")
        screenshot = QPushButton("截图识别")
        actions.addWidget(save)
        actions.addWidget(clear)
        actions.addWidget(screenshot)
        layout.addLayout(actions)
        layout.addStretch()
        save.clicked.connect(self.save_requested.emit)
        clear.clicked.connect(self.clear_requested.emit)
        screenshot.clicked.connect(self.screenshot_requested.emit)
        self._connect_dirty_signals()

    def _connect_dirty_signals(self) -> None:
        for selector in (
            self.comp, self.augments, self.components, self.completed, self.artifacts,
            self.radiant_items, self.emblems, self.units,
        ):
            selector.values_changed.connect(self.mark_dirty)
        self.stage.currentIndexChanged.connect(self.mark_dirty)
        for spinbox in (self.level, self.gold, self.hp):
            spinbox.valueChanged.connect(self.mark_dirty)
        self.notes.textChanged.connect(self.mark_dirty)

    def mark_dirty(self) -> None:
        if not self._dirty:
            self._dirty = True
            self.dirty_changed.emit(True)

    def game_state(self, question: str = "") -> GameState:
        selected_comp = self.comp.model.selected[0] if self.comp.model.selected else None
        return GameState(
            comp_name=self.comp.text().strip() or None,
            active_comp_id=(
                selected_comp.canonical_id
                if selected_comp and not selected_comp.canonical_id.startswith("legacy:")
                else None
            ),
            stage=self.stage.optional_value(),
            level=self.level.optional_value(),
            gold=self.gold.optional_value(),
            hp=self.hp.optional_value(),
            augments=_values(self.augments.text()),
            augment_choices=self._augment_choices,
            item_components=_values(self.components.text()),
            completed_items=_values(self.completed.text()),
            artifacts=_values(self.artifacts.text()),
            radiant_items=_values(self.radiant_items.text()),
            emblems=_values(self.emblems.text()),
            core_units=_values(self.units.text()),
            notes=self.notes.text().strip() or None,
            user_question=question or None,
        )

    def set_game_state(self, state: GameState) -> None:
        self._augment_choices = list(state.augment_choices)
        self.comp.setText(state.comp_name or "")
        self.stage.setCurrentText(state.stage or "未填写")
        self.level.setValue(state.level if state.level is not None else -1)
        self.gold.setValue(state.gold if state.gold is not None else -1)
        self.hp.setValue(state.hp if state.hp is not None else -1)
        self.augments.setText("，".join(state.augments))
        self.components.setText("，".join(state.item_components))
        self.completed.setText("，".join(state.completed_items))
        self.artifacts.setText("，".join(state.artifacts))
        self.radiant_items.setText("，".join(state.radiant_items))
        self.emblems.setText("，".join(state.emblems))
        self.units.setText("，".join(state.core_units))
        self.notes.setText(state.notes or "")
        self._dirty = False
        self.dirty_changed.emit(False)

    def clear_state(self) -> None:
        self.set_game_state(GameState())


def _values(text: str) -> list[str]:
    return [value.strip() for value in text.replace("，", ",").split(",") if value.strip()]
