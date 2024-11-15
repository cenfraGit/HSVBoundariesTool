# HSVBoundariesTool
# main.py
# 13/nov/2024
# cenfra


import wx
import cv2
import numpy as np
import json
from copy import copy, deepcopy
import os
import platform


if platform.system() == "Windows":
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    

def dip(*args):
    """Returns size using device independent pixels."""
    if len(args) == 1:
        return wx.ScreenDC().FromDIP(wx.Size(args[0], 0))[0]
    elif len(args) == 2:
        return wx.ScreenDC().FromDIP(wx.Size(args[0], args[1]))
    else:
        raise ValueError("DIP: Exceeded number of arguments.")
    

#hsvBounds = {"main": {"lower": (91, 175, 251), "upper": (179, 255, 255)}}
hsvBounds = {}
hsvEditLower = (0, 0, 0)
hsvEditUpper = (179, 255, 255)

activeMasks = []


class VariablePanel(wx.Panel):
    def __init__(self, parent, variableName, mainFrame, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.variableName = variableName
        self.mainFrame = mainFrame
        self.active = True

        # ---------------------- set up sizer ------------------------ #

        self.sizer = wx.GridBagSizer()
        self.SetSizer(self.sizer)

        self._checkbox = wx.CheckBox(self, label=self.variableName)
        self._checkbox.SetValue(1)
        self._buttonEdit = wx.Button(self, label="Edit...")
        self._buttonRemove = wx.Button(self, label="Remove")

        self._checkbox.Bind(wx.EVT_CHECKBOX, self._on_checkbox)
        self._buttonEdit.Bind(wx.EVT_BUTTON, self._on_edit)
        self._buttonRemove.Bind(wx.EVT_BUTTON, self._on_remove)

        self.sizer.Add(self._checkbox, pos=(0, 0), flag=wx.EXPAND|wx.ALL, border=dip(10))
        self.sizer.Add(self._buttonEdit, pos=(0, 1), flag=wx.ALL, border=dip(10))
        self.sizer.Add(self._buttonRemove, pos=(0, 2), flag=wx.ALL, border=dip(10))

        self.sizer.AddGrowableCol(0, 1)
        
        self.SetBackgroundColour(wx.YELLOW)


    def _on_edit(self, event):
        frame = EditVariableFrame(self, mode="edit", variableName=self.variableName, mainFrame=self.mainFrame)
        frame.Show()

    def _on_checkbox(self, event):
        global activeMasks
        if self._checkbox.GetValue() and self.variableName not in activeMasks:
            activeMasks.append(self.variableName)
        elif not self._checkbox.GetValue() and self.variableName in activeMasks:
            activeMasks.remove(self.variableName)
        print(activeMasks)


    def _on_remove(self, event):
        del hsvBounds[self.variableName]
        self.mainFrame._update_variable_panels()
        

class SourcePanel(wx.Panel):
    def __init__(self, parent, source=0, showMask=False, edit=False, img_size=(400, 300)):
        super().__init__(parent)
        self.source = source
        self.showMask = showMask
        self.capture = None
        self.image = None
        self.edit = edit # if in edit mode
        self.img_size = img_size
        self.timer = wx.Timer(self)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
        self.Bind(wx.EVT_TIMER, self.update_frame, self.timer)

        self.image_bitmap = wx.StaticBitmap(self, bitmap=wx.Bitmap(*dip(*img_size)))
        self.sizer.Add(self.image_bitmap, 1, wx.EXPAND)
        self.set_source(source)
        self.timer.Start(33)  # ~30 FPS


    def set_source(self, source):
        """Change the source dynamically (webcam number, image path, or video path)."""
        if self.capture and self.capture.isOpened():
            self.capture.release()

        self.source = source
        if isinstance(source, int):
            # Webcam source
            self.capture = cv2.VideoCapture(source)
        elif isinstance(source, str):
            if source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Image source
                self.capture = None
                self.image = cv2.imread(source)
                if self.image is not None:
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                    self.original_image = deepcopy(self.image)
                    if self.showMask:
                        self.image = self._combine_color_masks(self.image)
                
            else:
                # Video file source
                self.capture = cv2.VideoCapture(source)


    def update_frame(self, event):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                # Convert the color from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.showMask:
                    frame = self._combine_color_masks(frame)

                # Convert the frame to a wx.Image and display it
                h, w = frame.shape[:2]
                wx_image = wx.Image(w, h, frame.tobytes())
                width, height = dip(self.img_size[0], self.img_size[1])
                wx_image.Rescale(width, height)
                self.image_bitmap.SetBitmap(wx.Bitmap(wx_image))
            else:
                # If the source is a video and it reaches the end, loop it
                if isinstance(self.source, str) and not self.source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif self.image is not None:
            #frame = deepcopy(self.original_image)
            frame = self.original_image
            # Display the static image if the source is an image path
            if self.showMask:
                frame = self._combine_color_masks(self.original_image)
            h, w = frame.shape[:2]
            wx_image = wx.Image(w, h, frame.tobytes())
            width, height = dip(self.img_size[0], self.img_size[1])
            wx_image.Rescale(width, height)
            self.image_bitmap.SetBitmap(wx.Bitmap(wx_image))


    def _combine_color_masks(self, image_rgb):
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        masks = []

        if self.edit:  # if in edit mode, use edit variables
            global hsvEditLower, hsvEditUpper
            mask = cv2.inRange(image_hsv, hsvEditLower, hsvEditUpper)

            # Apply the mask to the original image to ensure it returns a 3-channel image
            result = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
            return result  # Return the masked RGB image

        # Generate masks based on the provided HSV bounds
        for color, bounds in hsvBounds.items():
            if color not in activeMasks:
                continue
            lower = np.array(bounds['lower'])
            upper = np.array(bounds['upper'])
            mask = cv2.inRange(image_hsv, lower, upper)
            #print("appended", color)
            masks.append(mask)

        if len(masks) == 0:
            return image_rgb  # No masks created, return the original image

        # Combine all masks into one
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        # combined_mask = cv2.bitwise_or(masks[0], masks[1])
        # for i in range(2, len(masks)):
        #     combined_mask = cv2.bitwise_or(combined_mask, masks[i])

        #print(len(masks), counter)



        # Apply the combined mask to the original image
        #result = cv2.bitwise_and(image_rgb, image_rgb, mask=combined_mask)
        result = cv2.bitwise_and(image_rgb, image_rgb, mask=combined_mask)

        return result


    def pause(self):
        """Pause the frame updates."""
        if self.timer.IsRunning():
            self.timer.Stop()
            print("Panel paused.")


    def resume(self):
        """Resume the frame updates."""
        if not self.timer.IsRunning():
            self.timer.Start(33)
            print("Panel resumed.")


class GradientPanel(wx.Panel):
    def __init__(self, parent, hue=0, gradient_type='saturation'):
        super().__init__(parent, size=(400, 30))
        self.gradient_type = gradient_type
        self.hue = hue  # Initial hue value

        # Bind paint event
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def set_hue(self, hue):
        self.hue = hue
        self.Refresh()  # Refresh the panel to trigger a repaint

    def on_paint(self, event):
        dc = wx.PaintDC(self)
        width, height = self.GetSize()

        # Create a gradient image based on the type
        if self.gradient_type == 'saturation':
            gradient = np.zeros((height, width, 3), dtype=np.uint8)
            for x in range(width):
                saturation = int((x / width) * 255)
                gradient[:, x, :] = [self.hue, saturation, 255]  # HSV with full value

        elif self.gradient_type == 'value':
            gradient = np.zeros((height, width, 3), dtype=np.uint8)
            for x in range(width):
                value = int((x / width) * 255)
                gradient[:, x, :] = [self.hue, 255, value]  # HSV with full saturation

        # Convert HSV to RGB for display
        rgb_gradient = cv2.cvtColor(gradient, cv2.COLOR_HSV2RGB)

        # Convert to wx.Bitmap
        image = wx.Image(width, height, rgb_gradient.tobytes())
        bitmap = wx.Bitmap(image)

        # Draw the bitmap
        dc.DrawBitmap(bitmap, 0, 0)


class HSVSliders(wx.Panel):
    def __init__(self, parent, lower=False, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        #self.hue = hue
        self.lower = lower
        
        # self.variableName = variableName
        # if self.variableName:
        #     global hsvEditLower, hsvEditUpper
        #     hsvEditLower = hsvBounds[self.variableName]["lower"]
        #     hsvEditUpper = hsvBounds[self.variableName]["upper"]
        # else:
        #     hsvEditLower = (0, 0, 0)
        #     hsvEditUpper = (179, 255, 255)

        if self.lower:
            hue = hsvEditLower[0]
            saturation = hsvEditLower[1]
            value = hsvEditLower[2]
        else:
            hue = hsvEditUpper[0]
            saturation = hsvEditUpper[1]
            value = hsvEditUpper[2]


        self.sizer = wx.GridBagSizer()
        self.SetSizer(self.sizer)

        self.color_preview = wx.Panel(self, size=dip(400, 50))
        self.color_preview.SetBackgroundColour(wx.Colour(255, 255, 255))

        # Load and scale the hue gradient image
        gradient_image = wx.Image('images/hue.png', wx.BITMAP_TYPE_PNG)
        gradient_image = gradient_image.Scale(*dip(400, 30))  # Match the slider size        

        # Create sliders for H, S, and V
        self.h_slider = wx.Slider(self, value=hue, minValue=0, maxValue=179, size=dip(400, -1))
        self.s_slider = wx.Slider(self, value=saturation, minValue=0, maxValue=255, size=dip(400, -1))
        self.v_slider = wx.Slider(self, value=value, minValue=0, maxValue=255, size=dip(400, -1))

        # Create dynamic gradient panels for saturation and value
        self.saturation_panel = GradientPanel(self, gradient_type='saturation')
        self.value_panel = GradientPanel(self, gradient_type='value')

        self.sizer.Add(wx.StaticText(self, label='Hue:'), pos=(0, 0))
        self.sizer.Add(wx.StaticBitmap(self, -1, wx.Bitmap(gradient_image)), pos=(1, 0), flag=0)
        self.sizer.Add(self.h_slider, pos=(2, 0), flag=0)

        self.sizer.Add(wx.StaticText(self, label='Saturation:'), pos=(3, 0))
        self.sizer.Add(self.saturation_panel, pos=(4, 0), flag=0)
        self.sizer.Add(self.s_slider, pos=(5, 0), flag=0)

        self.sizer.Add(wx.StaticText(self, label='Saturation:'), pos=(6, 0))
        self.sizer.Add(self.value_panel, pos=(7, 0), flag=0)
        self.sizer.Add(self.v_slider, pos=(8, 0), flag=0)
        self.sizer.Add(self.color_preview, pos=(9, 0), flag=0)

        self.sizer.AddGrowableCol(0, 1)

        self.sizer.Layout()

        # Bind slider events
        self.h_slider.Bind(wx.EVT_SLIDER, self.on_slider_change)
        self.s_slider.Bind(wx.EVT_SLIDER, self.on_slider_change)
        self.v_slider.Bind(wx.EVT_SLIDER, self.on_slider_change)


    def on_slider_change(self, event):
        global hsvEditLower, hsvEditUpper
        # Get current slider values
        hue = self.h_slider.GetValue()
        saturation = self.s_slider.GetValue()
        value = self.v_slider.GetValue()

        # Update gradient panels
        self.saturation_panel.set_hue(hue)
        self.value_panel.set_hue(hue)

        # Create an HSV color and convert it to RGB
        hsv_color = np.uint8([[[hue, saturation, value]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]

        # Update the color preview panel
        self.color_preview.SetBackgroundColour(wx.Colour(rgb_color[0], rgb_color[1], rgb_color[2]))
        self.color_preview.Refresh()

        if self.lower:
            hsvEditLower = hue, saturation, value
        else:
            hsvEditUpper = hue, saturation, value


    def GetHSV(self):
        hue = self.h_slider.GetValue()
        saturation = self.s_slider.GetValue()
        value = self.v_slider.GetValue()
        return (hue, saturation, value)


class EditVariableFrame(wx.Frame):
    def __init__(self, parent, mode="add", variableName=None, mainFrame=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.mode = mode
        self.variableName = variableName
        self.mainFrame = mainFrame

        #self.mainFrame.sourcePanel.pause()

        if self.mode == "edit":
            global hsvEditLower, hsvEditUpper
            hsvEditLower = tuple(hsvBounds[self.variableName]["lower"])
            hsvEditUpper = tuple(hsvBounds[self.variableName]["upper"])
        else:
            hsvEditLower = (0, 0, 0)
            hsvEditUpper = (179, 255, 255)


        self.Bind(wx.EVT_CLOSE, self._on_close)

        self._init_gui()

    def _init_gui(self):

        self.SetTitle(f"{self.mode.capitalize()} variable...")
        self.SetMinClientSize(dip(1200, 700))

        self.sizer = wx.GridBagSizer()
        self.SetSizer(self.sizer)

        self.sourcePanel = SourcePanel(self, self.mainFrame.sourcePanel.source, True, True, (600, 400))

        self.boundsPanel = wx.Panel(self)
        self.boundsPanelSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.boundsPanel.SetSizer(self.boundsPanelSizer)
        self.lowerPanel = HSVSliders(self.boundsPanel, lower=True)
        self.upperPanel = HSVSliders(self.boundsPanel, lower=False)
        self.boundsPanelSizer.Add(self.lowerPanel, 0, wx.EXPAND|wx.ALL, border=dip(5))
        self.boundsPanelSizer.Add(self.upperPanel, 0, wx.EXPAND|wx.ALL, border=dip(5))
        self.boundsPanelSizer.Layout()

        self.buttonsPanel = wx.Panel(self)
        self.buttonPanelSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.buttonsPanel.SetSizer(self.buttonPanelSizer)
        self.buttonOk = wx.Button(self.buttonsPanel, label="Ok")
        self.buttonCancel = wx.Button(self.buttonsPanel, label="Cancel")
        self.buttonPanelSizer.Add(self.buttonCancel, 0, flag=wx.ALL, border=dip(5))
        self.buttonPanelSizer.Add(self.buttonOk, 0, flag=wx.ALL, border=dip(5))
        self.buttonPanelSizer.Layout()
        
        self.sizer.Add(self.sourcePanel, pos=(0, 0), flag=wx.ALIGN_CENTER)
        self.sizer.Add(self.boundsPanel, pos=(1, 0), flag=wx.ALIGN_CENTER)
        self.sizer.Add(self.buttonsPanel, pos=(2, 0), flag=wx.ALIGN_RIGHT)

        self.sizer.AddGrowableCol(0, 1)
        self.sizer.AddGrowableRow(1, 1)
        self.sizer.Layout()

        self.buttonOk.Bind(wx.EVT_BUTTON, self._on_ok)
        self.buttonCancel.Bind(wx.EVT_BUTTON, self._on_cancel)

    def _on_close(self, event):
        #self.mainFrame.sourcePanel.resume()
        self.Close()

    def _on_ok(self, event):
        global hsvBounds, activeMasks
        if self.mode == "add":
            varName = ""
            while varName.strip() == "":
                dlg = wx.TextEntryDialog(self, "Enter variable name:", "VariableName", "")    
                dlg.ShowModal()
                varName = dlg.GetValue()
                activeMasks.append(varName)
        else:
            varName = self.variableName
        hsvBounds[varName] = {"lower": self.lowerPanel.GetHSV(), "upper": self.upperPanel.GetHSV()}
        
        self.mainFrame._update_variable_panels()

        self.sourcePanel.pause()
        self.Destroy()

    def _on_cancel(self, event):
        self.sourcePanel.pause()
        #self.Close()
        self.Destroy()


class MainFrame(wx.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self._start_ui()


    def _start_ui(self):
        
        self.SetTitle("HSV Color Range Tool")
        self.SetMinClientSize(dip(1600, 900))

        self._init_menubar()

        # ---------------------- main panel ------------------------ #

        self.mainPanel = wx.Panel(self)
        self.mainPanelSizer = wx.GridBagSizer()
        self.mainPanel.SetSizer(self.mainPanelSizer)
        self.mainPanel.SetBackgroundColour(wx.BLUE)


        # ---------------------- left panel ------------------------ #

        self.leftPanel = wx.Panel(self.mainPanel)
        self.leftPanel.SetBackgroundColour(wx.GREEN)
        self.leftPanelSizer = wx.BoxSizer(wx.VERTICAL)
        self.leftPanel.SetSizer(self.leftPanelSizer)
        self.scrolledPanel = wx.ScrolledWindow(self.leftPanel)
        self.scrolledPanel.SetScrollbars(20, 20, 55, 40)
        self.scrolledPanelSizer = wx.BoxSizer(wx.VERTICAL)
        self.scrolledPanel.SetSizer(self.scrolledPanelSizer)
        self._update_variable_panels()
        self.leftPanelSizer.Add(self.scrolledPanel, 1, flag=wx.EXPAND)
        self.leftPanelSizer.Layout()

        # ---------------------- right panel ------------------------ #

        self.rightPanel = wx.Panel(self.mainPanel)
        self.rightPanel.SetBackgroundColour(wx.CYAN)
        self.rightPanelSizer = wx.BoxSizer(wx.VERTICAL)
        self.rightPanel.SetSizer(self.rightPanelSizer)
        #self.sourcePanel = SourcePanel(self.rightPanel, source="test.mp4", showMask=True, img_size=(1100, 900))
        self.sourcePanel = SourcePanel(self.rightPanel, source="images/hue.png", showMask=True, img_size=(1100, 900))
        self.rightPanelSizer.Add(self.sourcePanel, 1, wx.EXPAND)

        #self.rightPanelSizer.Add(wx.RadioButton(self.rightPanel, label="test"), 1, wx.EXPAND)

        # ---------------------- add panels to sizer ------------------------ #

        self.mainPanelSizer.Add(self.leftPanel, pos=(0, 0), flag=wx.EXPAND)
        self.mainPanelSizer.Add(self.rightPanel, pos=(0, 1), flag=wx.EXPAND)
        self.mainPanelSizer.AddGrowableCol(0, 1)
        # self.mainPanelSizer.AddGrowableCol(1, 1)
        self.mainPanelSizer.AddGrowableRow(0, 1)

        self.mainPanelSizer.Layout()

        self.Bind(wx.EVT_CLOSE, self._on_close)


    def _init_menubar(self):
        self.menubar = wx.MenuBar()

        fileMenu = wx.Menu()
        fileMenu.Append(101, "Open JSON", "")
        fileMenu.AppendSeparator()
        fileMenu.Append(102, "Save As...", "")

        variablesMenu = wx.Menu()
        variablesMenu.Append(103, "Add variable", "")

        self.menubar.Append(fileMenu, "File")

        self.menubar.Append(variablesMenu, "Variables")

        self.SetMenuBar(self.menubar)

        self.Bind(wx.EVT_MENU, self._on_open_file, id=101)
        self.Bind(wx.EVT_MENU, self._on_save_as, id=102)

        self.Bind(wx.EVT_MENU, self._on_add_variable, id=103)


    def _on_open_file(self, event):
        global hsvBounds, activeMasks
        dlg = wx.FileDialog(self)
        dlg.ShowModal()
        with open(dlg.GetPath(), 'r', encoding='utf-8') as file:
            data = json.load(file)
        hsvBounds = data
        for key in hsvBounds.keys():
            activeMasks.append(key)
        self._update_variable_panels()


    def _on_save_as(self, event):
        dlg = wx.FileDialog(self)
        dlg.ShowModal()
        with open(dlg.GetPath(), "w", encoding="utf-8") as file:
            json.dump(hsvBounds, file)


    def _update_variable_panels(self):
        # delete existing panels
        for item in self.scrolledPanelSizer.GetChildren():
            window = item.GetWindow()
            if window:
                self.scrolledPanelSizer.Detach(window)
                window.Destroy()
        # display panels
        for item in hsvBounds:
            window = VariablePanel(self.scrolledPanel, variableName=item, mainFrame=self)
            self.scrolledPanelSizer.Add(window, 0, wx.EXPAND)
        self.scrolledPanelSizer.Layout()
        

    def _on_add_variable(self, event):
        frame = EditVariableFrame(self, "add", "", self)
        frame.Show()
        #self._update_variable_panels()

    def _on_close(self, event):
        self.sourcePanel.pause()
        self.Destroy()
    

if __name__ == "__main__":
    app = wx.App()
    frame = MainFrame(None)
    frame.Show()
    app.MainLoop()
