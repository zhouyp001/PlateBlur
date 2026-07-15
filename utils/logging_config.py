import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_FMT = '%(asctime)s - [%(threadName)s] - %(name)s - %(levelname)s - %(message)s'
LOG_DATE = '%H:%M:%S'
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5MB
LOG_BACKUP_COUNT = 3


def _is_console_available():
    """打包模式（--windows-disable-console）下 sys.stderr 可能不可用"""
    try:
        sys.stderr.fileno()
        return True
    except Exception:
        return False


def _get_log_dir():
    localappdata = os.environ.get('LOCALAPPDATA')
    if localappdata:
        return Path(localappdata) / 'hide-license'
    return Path.home() / '.hide-license' / 'logs'


def setup_logging():
    log_dir = _get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'app.log'

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # 文件 handler（始终启用）
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8',
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FMT, LOG_DATE))
    root.addHandler(file_handler)

    # 控制台 handler（仅开发模式）
    if _is_console_available():
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(LOG_FMT, LOG_DATE))
        root.addHandler(console_handler)

    logging.getLogger(__name__).info("日志系统初始化完成: %s", log_file)
