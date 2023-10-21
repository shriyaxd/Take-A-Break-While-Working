import tkinter as tk
from PIL import Image, ImageTk
from itertools import count

import os
import random

path = 'gifs'
images = []
fileNames = os.listdir(path)

def show_gif():
    
    class ImageLabel(tk.Label):
        """a label that displays images, and plays them if they are gifs"""
        def load(self, im):
            if isinstance(im, str):
                im = Image.open(im)
            self.loc = 0
            self.frames = []

            try:
                for i in count(1):
                    self.frames.append(ImageTk.PhotoImage(im.copy()))
                    im.seek(i)
            except EOFError:
                pass

            try:
                self.delay = im.info['duration']
            except:
                self.delay = 100

            if len(self.frames) == 1:
                self.config(image=self.frames[0])
            else:
                self.next_frame()

        def unload(self):
            self.config(image="")
            self.frames = None

        def next_frame(self):
            if self.frames:
                self.loc += 1
                self.loc %= len(self.frames)
                self.config(image=self.frames[self.loc])
                self.after(self.delay, self.next_frame)

    def close_tkinter():
        root.destroy()

    root = tk.Tk()
    lbl = ImageLabel(root)
    lbl.grid(row=1, column=0)
    text = tk.Label(root, text="Take a break")
    text.grid(row=0, column=0)
    lbl.load(f"gifs/{random.choice(fileNames)}")

    # Schedule the tkinter window to close after 5 seconds
    root.after(3000, close_tkinter)

    root.mainloop()

