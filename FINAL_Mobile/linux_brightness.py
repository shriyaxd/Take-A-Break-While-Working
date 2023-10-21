import os

def brightness_adjust(brightness):
    cmd=f'brightnessctl s {brightness}% >/dev/null 2>&1'
    os.system(cmd)

