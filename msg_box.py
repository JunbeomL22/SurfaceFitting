import tkinter as tk
from tkinter import messagebox

def show_message_box(msg, title):
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    messagebox.showinfo(title, msg)

