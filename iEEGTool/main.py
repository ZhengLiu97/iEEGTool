# -*- coding: UTF-8 -*-
'''
@Project ：iEEGTool 
@File    ：main.py
@Author  ：Barry
@Date    ：2022/2/18 2:21 
'''

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    from gui.main_win import MainWindow

    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    Window = MainWindow()
    # Window.showMaximized()
    Window.show()
    sys.exit(app.exec_())