from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QFileDialog, QLineEdit, QTextEdit, \
    QProgressBar, QLabel

from recognizer import recognize_by_DeepSpeech
import os


class AudioGUI():
    def __init__(self):
        self.app = QApplication([])
        self.window = QWidget()
        self.window.resize(800, 600)
        self.layout = QGridLayout()

        self.file_dlg = QFileDialog()
        self.file_dlg.setFileMode(QFileDialog.ExistingFile)

        # select audio file name
        self.audio_line = QLineEdit()
        self.layout.addWidget(self.audio_line, 0, 0, 1, 10)
        self.select_btn = QPushButton("select")
        self.layout.addWidget(self.select_btn, 0, 10, 1, 1)

        # exe btn
        self.exec_btn = QPushButton("exec")
        self.layout.addWidget(self.exec_btn, 0, 11, 1, 1)
        # clear btn
        self.clear_btn = QPushButton("clear")
        self.layout.addWidget(self.clear_btn, 0, 12, 1, 1)

        # result box
        self.result_edit = QTextEdit()
        self.layout.addWidget(self.result_edit, 1, 0, 5, 13)

        # progress bar
        self.pg_label = QLabel("running:")
        self.layout.addWidget(self.pg_label, 6, 0, 1, 1)
        self.pg_label.hide()
        self.pg_bar = QProgressBar()
        self.pg_bar.setValue(0)
        self.layout.addWidget(self.pg_bar, 6, 1, 1, 12)
        self.pg_bar.hide()

        # connect
        self.select_btn.clicked.connect(self.select_btn_click)
        self.file_dlg.fileSelected.connect(self.file_chosen)
        self.clear_btn.clicked.connect(self.clear_result_edit)
        self.exec_btn.clicked.connect(self.exec_task)

        self.window.setLayout(self.layout)
        self.window.show()

    def select_btn_click(self):
        self.file_dlg.show()

    def append_line(self, line):
        self.result_edit.setText(self.result_edit.toPlainText() + line + '\n')

    def exec_task(self):
        self.pg_bar.setValue(0)
        self.pg_bar.show()
        self.pg_label.show()

        # file name: self.audio_line.text()
        # get text: self.result_edit.toPlainText()
        # set text: self.result_edit.setText(str)
        # set progress bar value: self.pg_bar.setValue()
        filename = self.audio_line.text()
        if not os.path.isfile(filename):
            self.result_edit.setText('invalid filename!')
            return
        self.result_edit.setText('')
        self.pg_bar.setValue(0)
        recognize_by_DeepSpeech(filename, 'data/deepspeech-0.5.1-models', gui=self)

        self.pg_bar.hide()
        self.pg_label.hide()

    def file_chosen(self, filename):
        self.audio_line.setText(filename)

    def clear_result_edit(self):
        self.result_edit.clear()


def main():
    gui = AudioGUI()
    gui.app.exec_()


if __name__ == "__main__":
    main()
