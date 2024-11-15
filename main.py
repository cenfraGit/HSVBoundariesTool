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

from wx.lib.embeddedimage import PyEmbeddedImage

hue_image = PyEmbeddedImage(
    b'iVBORw0KGgoAAAANSUhEUgAAB1cAAAC1CAIAAACMHeExAAAAAXNSR0IArs4c6QAAAARnQU1B'
    b'AACxjwv8YQUAAAAJcEhZcwAADsQAAA7DAdpqmNwAACMNSURBVHhe7djT1nRdloXRkbbtzD9t'
    b'27Zt27Zt27Zt27ZtGzN3rV1txYovdmS+FUd10Gcb9/C02TPdC5PaX9v+UwdoO2DqQKkDtx0k'
    b'ddC2g6UOnjpE2yFTh2o7dOowqcO2HS51+NQR2o6YOlLbkVNHSR217Wipo7cdI3XM1LHajp06'
    b'TttxU8dL7dd2/NQJ2k6YOlHqxG0nSZ207WSpk6dO0XbK1KnaTp06Teq0badLnT51hrYzps7U'
    b'dubUWVJnbTtb6uxt50idM3WutnOnztN23tT5Uudvu0Dqgm0XSl04dZG2i6Yu1nbx1CVSl2y7'
    b'VOrSbZdJXTZ1ubbLp66QumLblVJXbrtK6qqpq7VdPXWNtmumrpW6dtt1Utdtu17q+qkbtN0w'
    b'daO2G6dukrpp281SN2+7ReqWqVu13Tp1m7bbpm6Xun3bHVJ3TN2p7c6pu7TdNXW31N3b7pG6'
    b'Z9u9UvdO3aftvqn7td0/9YDUA9selHpw20NSD009rO3hqUe0PTL1qNSj2x6TemzqcW2PTz2h'
    b'7YmpJ6We3PaU1FPbnpZ6euoZbc9MPavt2annpJ7b9rzU89tekHph6kVtL069pO2lqZelXt72'
    b'itQr216VenXqNW2vTb0u9fq2N6Te2Pam1JtTb2l7a+ptbW9PvSP1zrZ3pd7d9p7Ue1Pva3t/'
    b'6gNtH0x9KPXhto+kPtr2sdTHU59o+2TqU22fTn0m9dm2z6U+n/pC2xdTX2r7cuorqa+2fS31'
    b'9bZvpL6Z+lbbt1Pfaftu6nup77f9IPXDth+lfpz6SdtPUz9r+3nqF6lftv0q9eu236R+m/pd'
    b'2+9Tf0j9se1PqT+3/SX119Tf2v6e+kfbP1P/StX/7F+Vf1b+Ufl75W9tf638pfLnyp8qf6z8'
    b'oe33ld9Vflv5TeXXbb+q/LLyi8rPKz9r+2nlJ5UfV35U+WHlB23fr3yv8t3KdyrfbvtW5ZuV'
    b'b1S+Xvla5attX6l8ufKlyhcrX2j7fOVzlc9WPlP5dNunKp+sfKLy8crHKh9t+0jlw5UPVT5Y'
    b'+UDb+yvvq7y38p7Kuyvvantn5R2Vt1feVnlr21sqb668qfLGyhvaXl95XeW1lddUXl15Vdsr'
    b'K6+ovLzysspL215SeXHlRZUXVl7Q9vzK8yrPrTyn8uzKs9qeWXlG5emVp1We2vaUypMrT6o8'
    b'sfKEyuPbHld5bOUxlUdXHtX2yMojKg+vPKzy0LaHVB5ceVDlgZUHVO7fdr/KfSv3qdy7cq+2'
    b'e1buUbl75W6Vu1bu0nbnyp0qd6zcoXL7tttVblu5TeXWlVu13bJyi8rNKzer3LRyk7YbV25U'
    b'uWHlBpXrt12vct3KdSrXrlyrcs22a1SuXrla5aqVq7RduXKlyhUrV6hcvu1ylctWLlO5dOVS'
    b'lUu2XaJy8crFKhetXKTtwpULVS5YuUDl/JXztZ23cp7KuSvnqpyz7RyVs1fOVjlr5SxtZ66c'
    b'qXLGyhkqp6+cru20ldNUTl05VeWUbaeonLxysspJKyepnLjtRJUTVk5QOX5lv7bjVY5bOU7l'
    b'2JVjtR2zcozK0StHqxy1cpS2I1eOVDli5QiVw7cdrnLYymEqh64cqu2QlUNUDl45WOWglYO0'
    b'HbhyoMoBKweo7L9tfzU1u28u9xzvXu4p23O5e7anZs/Z7s2egz03ew72XOs52HOtN1O92enN'
    b'SPdCT3meC93zPLW553lu8xzmuc1zmOcqz2Geqzwnea7ynOTNHm/GuJd4ynAv8ZzhqcE9w3OD'
    b'pwD3Bs8Bnus7B3iu75zeub5zeje7uxjdzeLOuZ1a23M7t3YKbW/tHNqpsj20c2XnxM6V7Ymd'
    b'+jontvd1M66bZd3M6tzUKai9qXNQp5r2oPaaTimda9pTOnV0Tmnv6BTRuaM9opsF3czn3M4p'
    b'nL2dm+HcrGZP5tTLOZm9l1Ms5172WG6WcjOTcyPnTM6N3Azk9jpOadys45zGsY49jb2OYxp7'
    b'Hcc0znXcJ41zHf9rGucu7pPGuYtjGnsXxzT2LvY0jl2c07iXLk5R3KeLcxT36eIcxbGLPYpj'
    b'F3sUexd3i+JUxM0oTkXcJ4pzEccojkXsURyLOEdxnyLOUfyvRZxyuFnEKYf7FLHncCxiz2Ev'
    b'4pjDuYj75HAu4m45nFs45rC3cMxhb2HP4djCOYf7tHDO4T4tnEI4tHBbCPf5Xzef120hHN/W'
    b'/1MIpwr+/wzh3MIdQtg/0f8cwrmFYwjnFv7XEI5/5+bT+R9C2N/NzV9z89EcQzi3cAzh3MIx'
    b'hL2FYwjHz3IOYW9hD2Fv4eJDuflNbr6SYwh7C3sIewt7CHsLxxD2FvYQ9hb2EI4t3HwZF//F'
    b'zWexh3C9hWMOF4u4+TIufo2Lj+NiIHsj+wc5NbJncrGUPZZzL+dY9l4uJnPxoVz8KXs7ez7H'
    b'z7IXdDGivaOLKV38MntQF5s6ZrW/m4sfZ4/r2Nf+d46J7ZWdQztVdgztYms3f9CxuIvR7d3t'
    b'z+iY3l7fxZd0scE9w/03nTLcS9xjPJW4x3jxQ+1J7lWektyr3MM8vqqL3+rmw7r4sy52uqe6'
    b'13pKda/1YrDXXlgKTIEpMAWmwBSYAlNgCkyBKTAFpsAUeKccUmAKTIEp8P+2kAJTYApMgSkw'
    b'BabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUeFVcCkyBx6hSYApMgSkwBabAFJgCU2AK'
    b'TIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBR6eXwpMgSkwBabAFJgCU2AKTIEpMAWm'
    b'wBSYAlNgCjzluhebAlNgCkyBd3p7KTAFpsAUeLmOFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAF'
    b'psAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYArf2UmAKTIEpMAWmwBSYAlNgCkyBKXD/'
    b'NynwFMKhhdtCSIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWm'
    b'wBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAewshBabAFJgCU2AKTIEpMAVeLzMFpsAUmAJT'
    b'YApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAq+iSoEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApM'
    b'gSkwBabAFJgCU2AKTIEpMAWmwBR4eH4pMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSnwlOte'
    b'bApMgSkwBd7p7aXAFJgCU+DlOlJgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWm'
    b'wBSYAlNgCkyBKTAFpsAUmAJTYArc2kuBKTAFpsAUmAJTYApMgSkwBabA/d+kwFMIhxZuCyEF'
    b'psAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabA'
    b'FJgCU2AKTIEpMAWmwBSYAu8thBSYAlNgCkyBKTAFpsAUeL3MFJgCU2AKTIEpMAWmwBSYAlNg'
    b'CkyBKTAFpsAUmAJTYAq8iioFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyB'
    b'KTAFpsAUmAJT4OH5pcAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwFOue7EpMAWmwBR4p7eX'
    b'AlNgCkyBl+tIgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabA'
    b'FJgCU2AKTIEpcGsvBabAFJgCU2AKTIEpMAWmwBSYAvd/kwJPIRxauC2EFJgCU2AKTIEpMAWm'
    b'wBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAU'
    b'mAJTYAq8txBSYApMgSkwBabAFJgCU+D1MlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AK'
    b'TIEp8CqqFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIGH'
    b'55cCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJPue7FpsAUmAJT4J3eXgpMgSkwBV6uIwWm'
    b'wBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsCt'
    b'vRSYAlNgCkyBKTAFpsAUmAJTYArc/00KPIVwaOG2EFJgCkyBKTAFpsAUmAJTYApMgSkwBabA'
    b'FJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEp8N5CSIEp'
    b'MAWmwBSYAlNgCkyB18tMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsCrqFJgCkyB'
    b'KTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFHp5fCkyBKTAFpsAU'
    b'mAJTYApMgSkwBabAFJgCU2AKPOW6F5sCU2AKTIF3enspMAWmwBR4uY4UmAJTYApMgSkwBabA'
    b'FJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCt/ZSYApMgSkwBabA'
    b'FJgCU2AKTIEpcP83KfAUwqGF20JIgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAU'
    b'mAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsB7CyEFpsAUmAJTYApMgSkw'
    b'BV4vMwWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCr6JKgSkwBabAFJgCU2AKTIEp'
    b'MAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFHh4fikwBabAFJgCU2AKTIEpMAWmwBSY'
    b'AlNgCkyBKfCU615sCkyBKTAF3untpcAUmAJT4OU6UmAKTIEpMAWmwBSYAlNgCkyBKTAFpsAU'
    b'mAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCtzaS4EpMAWmwBSYAlNgCkyBKTAFpsD9'
    b'36TAUwiHFm4LIQWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSY'
    b'AlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgC7y2EFJgCU2AKTIEpMAWmwBR4vcwUmAJTYApM'
    b'gSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCryKKgWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkw'
    b'BabAFJgCU2AKTIEpMAWmwBSYAlPg4fmlwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAU657'
    b'sSkwBabAFHint5cCU2AKTIGX60iBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSY'
    b'AlNgCkyBKTAFpsAUmAJTYApMgSlway8FpsAUmAJTYApMgSkwBabAFJgC93+TAk8hHFq4LYQU'
    b'mAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgC'
    b'U2AKTIEpMAWmwBSYAlNgCry3EFJgCkyBKTAFpsAUmAJT4PUyU2AKTIEpMAWmwBSYAlNgCkyB'
    b'KTAFpsAUmAJTYApMgSnwKqoUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAF'
    b'psAUmAJTYApMgYfnlwJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAk+57sWmwBSYAlPgnd5e'
    b'CkyBKTAFXq4jBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgC'
    b'U2AKTIEpMAWmwK29FJgCU2AKTIEpMAWmwBSYAlNgCtz/TQo8hXBo4bYQUmAKTIEpMAWmwBSY'
    b'AlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJT'
    b'YApMgSnw3kJIgSkwBabAFJgCU2AKTIHXy0yBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEp'
    b'MAWmwKuoUmAKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAUe'
    b'nl8KTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYAo85boXmwJTYApMgXd6eykwBabAFHi5jhSY'
    b'AlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAK3'
    b'9lJgCkyBKTAFpsAUmAJTYApMgSlw/zcp8BTCoYXbQkiBKTAFpsAUmAJTYApMgSkwBabAFJgC'
    b'U2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwHsLIQWm'
    b'wBSYAlNgCkyBKTAFXi8zBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAKvokqBKTAF'
    b'psAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUeHh+KTAFpsAUmAJT'
    b'YApMgSkwBabAFJgCU2AKTIEp8JTrXmwKTIEpMAXe6e2lwBSYAlPg5TpSYApMgSkwBabAFJgC'
    b'U2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AK3NpLgSkwBabAFJgC'
    b'U2AKTIEpMAWmwP3fpMBTCIcWbgshBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJT'
    b'YApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmALvLYQUmAJTYApMgSkwBabA'
    b'FHi9zBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKvIoqBabAFJgCU2AKTIEpMAWm'
    b'wBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU+Dh+aXAFJgCU2AKTIEpMAWmwBSYAlNg'
    b'CkyBKTAFpsBTrnuxKTAFpsAUeKe3lwJTYApMgZfrSIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJT'
    b'YApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKXBrLwWmwBSYAlNgCkyBKTAFpsAUmAL3'
    b'f5MCTyEcWrgthBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNg'
    b'CkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKvLcQUmAKTIEpMAWmwBSYAlPg9TJTYApMgSkw'
    b'BabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKfAqqhSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabA'
    b'FJgCU2AKTIEpMAWmwBSYAlNgCkyBh+eXAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCT7nu'
    b'xabAFJgCU+Cd3l4KTIEpMAVeriMFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNg'
    b'CkyBKTAFpsAUmAJTYApMgSkwBabArb0UmAJTYApMgSkwBabAFJgCU2AK3P9NCjyFcGjhthBS'
    b'YApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AK'
    b'TIEpMAWmwBSYAlNgCkyBKfDeQkiBKTAFpsAUmAJTYApMgdfLTIEpMAWmwBSYAlNgCkyBKTAF'
    b'psAUmAJTYApMgSkwBabAq6hSYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAU'
    b'mAJTYApMgSkwBR6eXwpMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCjzluhebAlNgCkyBd3p7'
    b'KTAFpsAUeLmOFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AK'
    b'TIEpMAWmwBSYArf2UmAKTIEpMAWmwBSYAlNgCkyBKXD/NynwFMKhhdtCSIEpMAWmwBSYAlNg'
    b'CkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApM'
    b'gSkwBabAewshBabAFJgCU2AKTIEpMAVeLzMFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWm'
    b'wBSYAq+iSoEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBR4'
    b'eH4pMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSnwlOtebApMgSkwBd7p7aXAFJgCU+DlOlJg'
    b'CkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYArc'
    b'2kuBKTAFpsAUmAJTYApMgSkwBabA/d+kwFMIhxZuCyEFpsAUmAJTYApMgSkwBabAFJgCU2AK'
    b'TIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAu8thBSY'
    b'AlNgCkyBKTAFpsAUeL3MFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYAq8iioFpsAU'
    b'mAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJT4OH5pcAUmAJTYApM'
    b'gSkwBabAFJgCU2AKTIEpMAWmwFOue7EpMAWmwBR4p7eXAlNgCkyBl+tIgSkwBabAFJgCU2AK'
    b'TIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpcGsvBabAFJgCU2AK'
    b'TIEpMAWmwBSYAvd/kwJPIRxauC2EFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApM'
    b'gSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYAq8txBSYApMgSkwBabAFJgC'
    b'U+D1MlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEp8CqqFJgCU2AKTIEpMAWmwBSY'
    b'AlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIGH55cCU2AKTIEpMAWmwBSYAlNgCkyB'
    b'KTAFpsAUmAJPue7FpsAUmAJT4J3eXgpMgSkwBV6uIwWmwBSYAlNgCkyBKTAFpsAUmAJTYApM'
    b'gSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsCtvRSYAlNgCkyBKTAFpsAUmAJTYArc'
    b'/00KPIVwaOG2EFJgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyB'
    b'KTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEp8N5CSIEpMAWmwBSYAlNgCkyB18tMgSkwBabA'
    b'FJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsCrqFJgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgC'
    b'U2AKTIEpMAWmwBSYAlNgCkyBKTAFHp5fCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKPOW6'
    b'F5sCU2AKTIF3enspMAWmwBR4uY4UmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyB'
    b'KTAFpsAUmAJTYApMgSkwBabAFJgCt/ZSYApMgSkwBabAFJgCU2AKTIEpcP83KfAUwqGF20JI'
    b'gSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEp'
    b'MAWmwBSYAlNgCkyBKTAFpsB7CyEFpsAUmAJTYApMgSkwBV4vMwWmwBSYAlNgCkyBKTAFpsAU'
    b'mAJTYApMgSkwBabAFJgCr6JKgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJT'
    b'YApMgSkwBabAFHh4fikwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKfCU615sCkyBKTAF3unt'
    b'pcAUmAJT4OU6UmAKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEp'
    b'MAWmwBSYAlNgCtzaS4EpMAWmwBSYAlNgCkyBKTAFpsD936TAUwiHFm4LIQWmwBSYAlNgCkyB'
    b'KTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkw'
    b'BabAFJgC7y2EFJgCU2AKTIEpMAWmwBR4vcwUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSY'
    b'AlNgCryKKgWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlPg'
    b'4fmlwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAU657sSkwBabAFHint5cCU2AKTIGX60iB'
    b'KTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSlw'
    b'ay8FpsAUmAJTYApMgSkwBabAFJgC93+TAk8hHFq4LYQUmAJTYApMgSkwBabAFJgCU2AKTIEp'
    b'MAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCry3EFJg'
    b'CkyBKTAFpsAUmAJT4PUyU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSnwKqoUmAJT'
    b'YApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgYfnlwJTYApMgSkw'
    b'BabAFJgCU2AKTIEpMAWmwBSYAk+57sWmwBSYAlPgnd5eCkyBKTAFXq4jBabAFJgCU2AKTIEp'
    b'MAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwK29FJgCU2AKTIEp'
    b'MAWmwBSYAlNgCtz/TQo8hXBo4bYQUmAKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkw'
    b'BabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSnw3kJIgSkwBabAFJgCU2AK'
    b'TIHXy0yBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwKuoUmAKTIEpMAWmwBSYAlNg'
    b'CkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAUenl8KTIEpMAWmwBSYAlNgCkyBKTAF'
    b'psAUmAJTYAo85boXmwJTYApMgXd6eykwBabAFHi5jhSYAlNgCkyBKTAFpsAUmAJTYApMgSkw'
    b'BabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAK39lJgCkyBKTAFpsAUmAJTYApMgSlw'
    b'/zcp8BTCoYXbQkiBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAF'
    b'psAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWmwHsLIQWmwBSYAlNgCkyBKTAFXi8zBabAFJgC'
    b'U2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAKvokqBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AK'
    b'TIEpMAWmwBSYAlNgCkyBKTAFpsAUeHh+KTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEp8JTr'
    b'XmwKTIEpMAXe6e2lwBSYAlPg5TpSYApMgSkwBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAF'
    b'psAUmAJTYApMgSkwBabAFJgCU2AK3NpLgSkwBabAFJgCU2AKTIEpMAWmwP3fpMBTCIcWbgsh'
    b'BabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApMgSkwBabAFJgCU2AKTIEpMAWm'
    b'wBSYAlNgCkyBKTAFpsAUmALvLYQUmAJTYApMgSkwBabAFHi9zBSYAlNgCkyBKTAFpsAUmAJT'
    b'YApMgSkwBabAFJgCU2AKvIoqBabAFJgCU2AKTIEpMAWmwBSYAlNgCkyBKTAFpsAUmAJTYApM'
    b'gSkwBabAFJgCU+Dh+aXAFJgCU2AKTIEpMAWmwBSYAlNgCkyBh2Znv38DafWwyFC5X1UAAAAA'
    b'SUVORK5CYII=')




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
        
        #self.SetBackgroundColour(wx.YELLOW)
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_SIZE, self._on_size)


    def _on_paint(self, event):
        dc = wx.PaintDC(self)
        dc.Clear()

        rect = self.GetClientRect()

        dc.SetPen(wx.GREY_PEN)
        #dc.SetBrush(wx.RED_BRUSH)
        dc.SetBrush(wx.TRANSPARENT_BRUSH)

        dc.DrawRectangle(rect)


    def _on_size(self, event):
        self.Refresh()
        self.sizer.Layout()


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
        #gradient_image = wx.Image('images/hue.png', wx.BITMAP_TYPE_PNG)
        gradient_image = wx.Image(hue_image.GetImage())
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
        self.SetMinClientSize(dip(1200, 750))
        self.SetBackgroundColour(wx.WHITE)

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
        self.Destroy()


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

        # ---------------------- left panel ------------------------ #

        self.leftPanel = wx.Panel(self.mainPanel)
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
        self.rightPanelSizer = wx.BoxSizer(wx.VERTICAL)
        self.rightPanel.SetSizer(self.rightPanelSizer)
        self.sourcePanel = SourcePanel(self.rightPanel, source="shapes.png", showMask=True, img_size=(1100, 900))
        #self.sourcePanel = SourcePanel(self.rightPanel, source="images/hue.png", showMask=True, img_size=(1100, 900))
        self.rightPanelSizer.Add(self.sourcePanel, 1, wx.EXPAND)

        # ---------------------- add panels to sizer ------------------------ #

        self.mainPanelSizer.Add(self.leftPanel, pos=(0, 0), flag=wx.EXPAND)
        self.mainPanelSizer.Add(self.rightPanel, pos=(0, 1), flag=wx.EXPAND)
        self.mainPanelSizer.AddGrowableCol(0, 1)
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
