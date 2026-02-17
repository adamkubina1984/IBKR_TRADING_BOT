from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QPlainTextEdit


class LogConsole(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)

    def append_line(self, text: str):
        if not text:
            return
        # appendPlainText automaticky přidá nový řádek
        self.appendPlainText(text)
        # posun na konec (když chceš, můžeš vynechat – Tab 1 to stejně dělá po volání)
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)
