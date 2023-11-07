import pydirectinput as pdi
from pynput.keyboard import Key, Listener
from ctypes import *
import PyHook3
import pythoncom
import win32api

flag = False


# def on_press(key):
#     global flag
#     try:
#         if not flag and key == Key.f2:
#             print('script started!')
#             flag = True
#
#         if flag:
#             if key.char == '1':
#                 pdi.press('j')
#             if key.char == '2':
#                 pdi.press('k')
#
#     except AttributeError:
#         pass
#
#
# def on_release(key):
#     if flag and key == Key.esc:
#         # Stop listener
#         print('script ended!')
#         return False


def onKeyboardEvent(event):
    # 监听键盘事件
    windowTitle = create_string_buffer(512)
    windll.user32.GetWindowTextA(event.Window, byref(windowTitle), 512)
    windowName = windowTitle.value.decode('gbk')
    if event.Ascii == 49:
        pdi.press('j')
    if event.Ascii == 50:
        pdi.press('k')
    if event.Ascii == 27:
        win32api.PostQuitMessage(0)
    print("当前您处于%s窗口" % windowName)
    print("当前刚刚按下了%s键" % str(event.Ascii))
    return True


def main():
    # 创建一个“钩子”管理对象
    hm = PyHook3.HookManager()
    # 监听所有键盘事件
    hm.KeyDown = onKeyboardEvent
    # 设置键盘“钩子”
    hm.HookKeyboard()
    # 进入循环，如不手动关闭，程序将一直处于监听状态
    pythoncom.PumpMessages()


if __name__ == "__main__":
    main()
