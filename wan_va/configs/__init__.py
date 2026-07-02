# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_robotwin_cfg import va_robotwin_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_libero_cfg import va_libero_cfg
from .va_libero_train_cfg import va_libero_train_cfg
from .va_libero_i2va import va_libero_i2va_cfg

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'libero': va_libero_cfg,
    'libero_train': va_libero_train_cfg,
    'libero_i2av': va_libero_i2va_cfg,
}
