Basic PC Specs:
    OS: Linux Mint 22.3 x86_64
    Host: 83J3 (IdeaPad Slim 5 15ARP10)
    Kernel: Linux 7.0.0-30-generic
    Packages: 2303 (dpkg), 10 (flatpak)
    Shell: bash 5.2.21
    DE: Cinnamon 6.6.9
    WM: Muffin (X11)
    Terminal: GNOME Terminal 3.52CPU: AMD Ryzen 7 7735HS (16) @ 4.83 GHz
    GPU: AMD Radeon 680M [Integrated]
To start, we're gonna use a python virtual environment (venv) because:
using pip install would require us to run this "pip install numpy matplotlib --break-system-packages" and it can cause conflicts with my linux system package manager if "apt" ever tries to update my python packages in future.


