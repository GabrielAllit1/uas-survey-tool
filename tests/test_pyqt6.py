import logging
import sys
from PyQt6.QtWidgets import QApplication
logging.basicConfig(level=logging.DEBUG, filename='test_pyqt6.log')
logger = logging.getLogger(__name__)
logger.info("Attempting to import PyQt6")
app = QApplication(sys.argv)
logger.info("QApplication initialized successfully")