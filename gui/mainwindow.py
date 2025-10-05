import os

import soundfile
import numpy as np
import urllib

from typing import List
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from slicer2 import Slicer

from gui.Ui_MainWindow import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButtonAddFiles.clicked.connect(self._q_add_audio_files)
        self.ui.pushButtonBrowse.clicked.connect(self._q_browse_output_dir)
        self.ui.pushButtonClearList.clicked.connect(self._q_clear_audio_list)
        self.ui.pushButtonAbout.clicked.connect(self._q_about)
        self.ui.pushButtonStart.clicked.connect(self._q_start)

        self.ui.progressBar.setMinimum(0)
        self.ui.progressBar.setMaximum(100)
        self.ui.progressBar.setValue(0)
        self.ui.pushButtonStart.setDefault(True)

        validator = QRegularExpressionValidator(QRegularExpression(r"\d+"))
        self.ui.lineEditThreshold.setValidator(QDoubleValidator())
        self.ui.lineEditMinLen.setValidator(validator)
        self.ui.lineEditMinInterval.setValidator(validator)
        self.ui.lineEditHopSize.setValidator(validator)
        self.ui.lineEditMaxSilence.setValidator(validator)
        self.ui.lineEditSpkId.setValidator(validator)
        

        self.ui.listWidgetTaskList.setAlternatingRowColors(True)

        # State variables
        self.workers: list[QThread] = []
        self.workCount = 0
        self.workFinished = 0
        self.processing = False

        self.setWindowTitle(QApplication.applicationName())

        # Must set to accept drag and drop events
        self.setAcceptDrops(True)

        # Get available formats/extensions supported
        self.availableFormats = [str(formatExt).lower(
        ) for formatExt in soundfile.available_formats().keys()]
        # libsndfile supports Opus in Ogg container
        # .opus is a valid extension and recommended for Ogg Opus (see RFC 7845, Section 9)
        # append opus for convenience as tools like youtube-dl(p) extract to .opus by default
        self.availableFormats.append("opus")

        self.formatAllFilter = " ".join(
            [f"*.{formatExt}" for formatExt in self.availableFormats])
        self.formatIndividualFilter = ";;".join(
            [f"{formatExt} (*.{formatExt})" for formatExt in sorted(self.availableFormats)])

    def _q_browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, "Browse Output Directory", ".")
        if path != "":
            self.ui.lineEditOutputDir.setText(QDir.toNativeSeparators(path))

    def _q_add_audio_files(self):
        if self.processing:
            self.warningProcessNotFinished()
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self, 'Select Audio Files', ".", f'Audio ({self.formatAllFilter});;{self.formatIndividualFilter}')
        for path in paths:
            item = QListWidgetItem()
            item.setSizeHint(QSize(200, 24))
            item.setText(QFileInfo(path).fileName())
            # Save full path at custom role
            item.setData(Qt.ItemDataRole.UserRole + 1, path)
            self.ui.listWidgetTaskList.addItem(item)

    def _q_clear_audio_list(self):
        if self.processing:
            self.warningProcessNotFinished()
            return

        self.ui.listWidgetTaskList.clear()

    def _q_about(self):
        QMessageBox.information(
            self, "About", "Audio Slicer v1.3.0\nCopyright 2020-2024 OpenVPI Team")

    def _q_start(self):
        if self.processing:
            self.warningProcessNotFinished()
            return

        item_count = self.ui.listWidgetTaskList.count()
        if item_count == 0:
            return

        output_format = self.ui.buttonGroup.checkedButton().text()
        if output_format == "mp3":
            ret = QMessageBox.warning(self, "Warning",
                                      "MP3 is not recommended for saving vocals as it is lossy. "
                                      "If you want to save disk space, consider using FLAC instead. "
                                      "Do you want to continue?",
                                      QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel)
            if ret == QMessageBox.Cancel:
                return

        class WorkThread(QThread):
            oneFinished = Signal()

            def __init__(self, filenames: List[str], window: MainWindow):
                super().__init__()

                self.filenames = filenames
                self.win = window

            def run(self):
                for filename in self.filenames:
                    audio, sr = soundfile.read(filename, dtype=np.float32)
                    is_mono = True
                    if len(audio.shape) > 1:
                        is_mono = False
                        audio = audio.T
                    slicer = Slicer(
                        sr=sr,
                        threshold=float(self.win.ui.lineEditThreshold.text()),
                        min_length=int(self.win.ui.lineEditMinLen.text()),
                        min_interval=int(
                            self.win.ui.lineEditMinInterval.text()),
                        hop_size=int(self.win.ui.lineEditHopSize.text()),
                        max_sil_kept=int(self.win.ui.lineEditMaxSilence.text()),
                        speaker_id=int(self.win.ui.lineEditSpkId.text())
                    )
                    chunks = slicer.slice(audio)
                    '''
                    out_dir = self.win.ui.lineEditOutputDir.text()
                    if out_dir == '':
                        out_dir = os.path.dirname(os.path.abspath(filename))
                    else:
                        # Make dir if not exists
                        info = QDir(out_dir)
                        if not info.exists():
                            info.mkpath(out_dir)
                    '''

                    #add libtts format
                    out_dir = self.win.ui.lineEditOutputDir.text()
                    speaker_id=int(self.win.ui.lineEditSpkId.text())
                    fn=os.path.basename(filename)
                    fn = fn.split(".")[0]
                    out_dir = os.path.join(out_dir, str(speaker_id), fn)
                    info = QDir(out_dir)
                    info.mkpath(out_dir)

                    ext = self.win.ui.buttonGroup.checkedButton().text()
                    for i, chunk in enumerate(chunks):
                        #path = os.path.join(out_dir, f'%s_%03d.{ext}' % (os.path.basename(filename)
                        #                                               .rsplit('.', maxsplit=1)[0], i))
                        path = os.path.join(out_dir, f'%s_%s_%03d.{ext}' % (speaker_id, fn, i))
                        if not is_mono:
                            chunk = chunk.T
                        soundfile.write(path, chunk, sr)

                    self.oneFinished.emit()

        # Collect paths
        paths: list[str] = []
        for i in range(0, item_count):
            item = self.ui.listWidgetTaskList.item(i)
            path = item.data(Qt.ItemDataRole.UserRole + 1)  # Get full path
            paths.append(path)

        self.ui.progressBar.setMaximum(item_count)
        self.ui.progressBar.setValue(0)

        self.workCount = item_count
        self.workFinished = 0
        self.setProcessing(True)

        # Start work thread
        worker = WorkThread(paths, self)
        worker.oneFinished.connect(self._q_oneFinished)
        worker.finished.connect(self._q_threadFinished)
        worker.start()

        self.workers.append(worker)  # Collect in case of auto deletion

    def _q_oneFinished(self):
        self.workFinished += 1
        self.ui.progressBar.setValue(self.workFinished)

    def _q_threadFinished(self):
        # Join all workers
        for worker in self.workers:
            worker.wait()
        self.workers.clear()
        self.setProcessing(False)

        QMessageBox.information(
            self, QApplication.applicationName(), "Slicing complete!")

    def warningProcessNotFinished(self):
        QMessageBox.warning(self, QApplication.applicationName(),
                            "Please wait for slicing to complete!")

    def setProcessing(self, processing: bool):
        enabled = not processing
        self.ui.pushButtonStart.setText(
            "Slicing..." if processing else "Start")
        self.ui.pushButtonStart.setEnabled(enabled)
        self.ui.pushButtonAddFiles.setEnabled(enabled)
        self.ui.listWidgetTaskList.setEnabled(enabled)
        self.ui.pushButtonClearList.setEnabled(enabled)
        self.ui.lineEditThreshold.setEnabled(enabled)
        self.ui.lineEditMinLen.setEnabled(enabled)
        self.ui.lineEditMinInterval.setEnabled(enabled)
        self.ui.lineEditHopSize.setEnabled(enabled)
        self.ui.lineEditMaxSilence.setEnabled(enabled)
        self.ui.lineEditSpkId.setEnabled(enabled)
        self.ui.lineEditOutputDir.setEnabled(enabled)
        self.ui.pushButtonBrowse.setEnabled(enabled)
        self.processing = processing

    # Event Handlers
    def closeEvent(self, event):
        if self.processing:
            self.warningProcessNotFinished()
            event.ignore()

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        valid = False
        for url in urls:
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            ext = os.path.splitext(path)[1]
            if ext[1:].lower() in self.availableFormats:
                valid = True
                break
        if valid:
            event.accept()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            ext = os.path.splitext(path)[1]
            if ext[1:].lower() not in self.availableFormats:
                continue
            item = QListWidgetItem()
            item.setSizeHint(QSize(200, 24))
            item.setText(QFileInfo(path).fileName())
            item.setData(Qt.ItemDataRole.UserRole + 1,
                         path)
            self.ui.listWidgetTaskList.addItem(item)
