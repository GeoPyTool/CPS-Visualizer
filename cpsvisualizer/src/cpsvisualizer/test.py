from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QListWidget, QDialog, QVBoxLayout

class DropDownList(QWidget):
    def __init__(self):
        super().__init__()

        self.button = QPushButton("Select an item", self)
        self.button.clicked.connect(self.show_list)

        self.list_widget = QListWidget()
        self.list_widget.addItems(["Item 1", "Item 2", "Item 3", "Item 4"])
        self.list_widget.itemClicked.connect(self.item_selected)

        layout = QVBoxLayout(self)
        layout.addWidget(self.button)

        self.setLayout(layout)

    def show_list(self):
        self.dialog = QDialog(self)
        self.dialog.setWindowTitle("Select an item")
        dialog_layout = QVBoxLayout(self.dialog)
        dialog_layout.addWidget(self.list_widget)
        self.dialog.setLayout(dialog_layout)
        self.dialog.exec()

    def item_selected(self, item):
        self.button.setText(item.text())
        self.dialog.close()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = DropDownList()
    window.show()
    sys.exit(app.exec())