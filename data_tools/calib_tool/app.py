"""Điểm khởi chạy ChArUco Calibrator."""
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LOG = os.path.join(HERE, "charuco_error.log")


def _write_log(text: str):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(text + "\n" + "-" * 70 + "\n")
    except Exception:
        pass


def _native_box(title: str, text: str):
    """Hộp thoại báo lỗi không cần Qt (dùng khi chưa nạp được PySide6)."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)
            return
        except Exception:
            pass
    print(title, text, sep="\n", file=sys.stderr)


def _report(exc_type, exc, tb):
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    _write_log(text)
    msg = (f"Chương trình gặp lỗi:\n\n{exc_type.__name__}: {exc}\n\n"
           f"Chi tiết đã ghi vào:\n{LOG}\n\nHãy gửi nội dung file này để được hỗ trợ.")
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "ChArUco Calibrator — lỗi", msg)
            return
    except Exception:
        pass
    _native_box("ChArUco Calibrator — lỗi", msg)


def main():
    sys.excepthook = _report
    try:
        import matplotlib
        matplotlib.use("QtAgg")
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QPalette, QColor
    except Exception:
        _report(*sys.exc_info())
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("ChArUco Calibrator")
    app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(QPalette.Window, QColor("#f9f9f7"))
    pal.setColor(QPalette.Base, QColor("#ffffff"))
    pal.setColor(QPalette.AlternateBase, QColor("#f3f3f1"))
    pal.setColor(QPalette.Button, QColor("#f3f3f1"))
    pal.setColor(QPalette.Text, QColor("#0b0b0b"))
    pal.setColor(QPalette.WindowText, QColor("#0b0b0b"))
    pal.setColor(QPalette.ButtonText, QColor("#0b0b0b"))
    pal.setColor(QPalette.Highlight, QColor("#2a78d6"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)

    try:
        from gui import MainWindow
        w = MainWindow()
    except Exception:
        _report(*sys.exc_info())
        sys.exit(1)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
