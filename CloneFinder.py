#!/user/bin/python3
# -*- coding: utf-8 -*-
import random
from sys import exit

import os
import sys
import argparse
import socket
import random
import numpy as np
import pandas as pd
import cv2 as cv2


from progress.bar import IncrementalBar
import math
import time
import datetime as dt
from threading import Thread
import threading

import common_funct as cf
import fu2clonefinder as f2cf
from ca_config import *

def COS_dist_calc(frs, fre, chunk_X2, chunk_fX, all_Y2, all_fY, dpath, lready_rdm, flog=None, console=0):
    try:
        _ltemp_ = []
        for X2, fX in zip(chunk_X2, chunk_fX):
            # print(type(X2[0,0,0]), type(fX[0,0,0]))
            XY = np.real(np.fft.irfft2(fX * np.conj(all_fY), axes=(1, 2)))
            _ltemp_.append(XY / (((np.expand_dims(X2, 2) * all_Y2) + np.finfo(float).eps).transpose((2, 0, 1, 3))))#*1000000000).astype(np.int32))
        # print(_ltemp_[0][0,0,0])
        np.save(os.path.join(dpath, f'rdm-{frs}-{fre - 1}.npy'), np.array(_ltemp_))
        str_log = f'\nThe file << rdm-{frs}-{fre - 1}.npy >> is created'
        cf.print_log(str_log, flog=flog, console=console)
        lready_rdm.append((frs, fre))
    except Exception as e:
        str_log = f'ERROR!!! (COS_dist_calc) {frs}-{fre - 1} \n {str(e)}'
        cf.print_log(str_log, flog=flog, console=console)
        raise

def load_rdm(frs, fre, lim,  dpath,  flog=None, console=0):
    try:
        rdm = np.load(os.path.join(dpath, f'rdm-{frs}-{fre - 1}.npy'))
        # rdm[rdm < lim] = 0
        str_log = f'\nThe file << rdm-{frs}-{fre - 1}.npy>> has been read successfully!'
        cf.print_log(str_log, flog=flog, console=console)
        return rdm
    except Exception as e:
        str_log = f'ERROR!!! (load_rdm {frs}-{fre - 1}) \n {str(e)}'
        cf.print_log(str_log, flog=flog, console=console)
        raise # невозможно прочитать файл rdm

# def look2compare_calc(frs, fre, llook, crop_size,  dpath, flog=None, console=0):
#     try:
#         (wc, hc) = crop_size
#         look_X2, look_fX = f2cf.calc2compare4look(llook, (wc, hc), console=1, flog=flog)
#         np.save(os.path.join(dpath, f'look_fx-{frs}-{fre - 1}.npy'), look_fX)
#         np.save(os.path.join(dpath, f'look_x2-{frs}-{fre - 1}.npy'), look_X2)
#     except Exception as e:
#         str_log = f'ERROR!!! (look2compare_calc {frs}-{fre - 1}) \n {str(e)}'
#         cf.print_log(str_log, flog=flog, console=console)

def set_lim_bb(val):
    global lim_bb
    lim_bb = val #cv.getTrackbarPos('MIN', f'{self.main_app.path}')
    cv2.setTrackbarPos(TRACK_BB_NAME, MAIN_WINDOW_NAME, lim_bb)

def set_lim_rdm(val):
    global lim_rdm
    lim_rdm = val #cv.getTrackbarPos('MIN', f'{self.main_app.path}')
    cv2.setTrackbarPos(TRACK_COS_NAME, MAIN_WINDOW_NAME, lim_rdm)

def set_lim_ssq2pix(val):
    global lim_ssq2pix
    lim_ssq2pix= val #cv.getTrackbarPos('MIN', f'{self.main_app.path}')
    cv2.setTrackbarPos(TRACK_P2P_NAME, MAIN_WINDOW_NAME, lim_ssq2pix)

def set_lim_h(val):
    global lim_h
    lim_h = val #cv.getTrackbarPos('MIN', f'{self.main_app.path}')
    cv2.setTrackbarPos(TRACK_HAM_NAME, MAIN_WINDOW_NAME, lim_h)

if __name__ == '__main__':

    # START -------------------  START -------------------- START -------------------- START -------------------- START
    try:
        try:
            ap = argparse.ArgumentParser(description='Parameters of the clone search utility')

            ap.add_argument("-fin", "--fin", type=str, default=None,
                            help="Path to video file")
            ap.add_argument("-flin", "--flin", type=str, default=None,
                            help="Path to video file")

            ap.add_argument("-lf", "--lf", nargs=2,  default=None,
                            help="Frame number and range for LOOK area")

            ap.add_argument("-lr", "--lr", nargs=4,  default=None,
                            help="Coordinates of the rectangular area in which the cloned fragments are searched "
                                 " If the parameter is not set, the search is performed over the entire area of the frame. "
                                 "Format: x_left, y_top, x_right, y_bottom")

            ap.add_argument("-sf", "--sf", default=None,
                            help="The SRC frame number")

            ap.add_argument("-vsf", "--vsf", default=0,
                            help="Frames variability of the SRC")

            ap.add_argument("-sr", "--sr", nargs=4, default=None,
                            help="Coordinates of the clone source area")

            ap.add_argument("-vsr", "--vsr", default=0,
                            help="Coordinates variability of the SRC")

            ap.add_argument("-fout ", "--fout", type=str, default='clone-search.csv',
                            help="File name with the list of detected frames,  default = clone-search.csv")

            ap.add_argument("-data ", "--data", type=str, default='data',
                            help="Folder for calculated data,  default = <<Project FOLDER\Data\>>")

            ap.add_argument("-log ", "--log", type=str, default='clone-search.log',
                            help="Log-file name")

            ap.add_argument("-verbosity", "--verbosity", type=str, default='',
                            help="=DEBUG to display extended information")

            ap.add_argument("-rm", "--rm", default=1,
                            help="=Recalculate matrices")
            ap.add_argument("-progress", "--progress", type=int, default=10555, help="=Socket #, default =10555")
            #
            # ap.add_argument("-yrgb", "--yrgb", type=str, default='',
            #                 help="Transform the color space: Y or R, or G, or B")
            # ap.add_argument("-scale", "--scale", type=int, default=0,
            #                 help="Scale: 2 or 4")
            # ap.add_argument("-contr", "--contr", type=int, default=0,
            #                 help="THRESH_BINARY in %: 1 to 100")
            # ap.add_argument("-uf", "--uf", nargs='+', type=str, default='',
            #                 help="User functions:func_reflect, func_rot90, func_rot180, func_rot270, func_inv")

            args = vars(ap.parse_args())
        except Exception as e:
            print(e)
            raise
            exit(222)

        file_log = args['log']
        flog = None
        try:
            abs_path_file_log = cf.take_path(file_log)
            if abs_path_file_log is not None:
                flog = open(abs_path_file_log, "w")
                str_log = f'Log-file created: << {file_log} >>'
                cf.print_log(str_log, flog=flog, console=1)
            else:
                raise ValueError(f'The file  << {file_log} >>  for recording the log has not been created')
        except Exception as e:
            print(e)
            exit(105)  # Файл «log» не может быть создан

        ###################### verbosity
        debug = 1
        try:
            if args['verbosity'] == 'DEBUG':
                debug = 1
                str_log = f'The argument  -verbosity set in  <<DEBUG>>'
                cf.print_log(str_log, flog=flog, console=debug)
            elif args['verbosity'] == '':
                debug = 0
                str_log = f'The argument  -verbosity set in  <<NO DEBUG>>'
                cf.print_log(str_log, flog=flog, console=debug)
            else:
                raise Exception('E001')
        except Exception as e:
            str_log = f'{args["verbosity"]}\n{str(e)}'
            raise Exception('E001')
            # cf.print_log(flog, str_log, console=debug)
            # flog.close()
            # exit(115)  # Неверно задан параметр verbosity

        ###################### read_only
        try:
            if int(args['rm']) == 1 or args['rm'] is None:
                recalc = 1
                str_log = f'The argument  -rm set in  << recalculate_matrices >>'
                cf.print_log(str_log, flog=flog, console=debug)

            elif int(args['rm']) == 0:
                recalc = 0
                str_log = f'The argument  -rm set in  << no recalculate_matrices >>'
                cf.print_log(str_log, flog=flog, console=debug)

            else:
                raise Exception('E002')
        except Exception as e:
            str_log = f'"{args["rm"]}"\n{str(e)}'
            raise Exception('E002')
            # str_log = f'ERROR!!! (recalculate_matrices) \nThe argument  -rm =  {args["rm"]} is not true \n{str(e)}'
            # cf.print_log(flog, str_log, console=debug)
            # flog.close()
            # exit(122)  # Неверно задан параметр recalculate_matrices

        ####################################################################### fin
        vfile, vs = None, None
        try:
            vfile, vs = cf.checking_vfile(args['fin'], par_name='fin', flog=flog, console=debug)
        except Exception as e:
            str_log = f'"{args["fin"]}"\n{str(e)}'
            raise Exception('E003')
            # cf.print_log(flog, str_log, console=debug)
            # flog.close()
            # exit(222)

        ####################################################################### flin
        vlfile, vl = None, None
        try:
            if args['flin'] is None:
                vlfile, vl = cf.checking_vfile(args['fin'], par_name='flin', flog=flog, console=debug)
            else:
                vlfile, vl = cf.checking_vfile(args['flin'], par_name='flin', flog=flog, console=debug)
        except Exception as e:
            str_log = f'"{args["flin"]}"\n{str(e)}'
            raise Exception('E004')
            # str_log = f'ERROR!!! (flin).\n{str(e)}'
            # cf.print_log(flog, str_log, console=debug)

            # flog.close()
            # exit(222)

        ####################################################################### lf
        try:
            if args['lf'] is not None:
                if isinstance(args['lf'], (list)) and len(args['lf']) == 2 \
                        and int(args['lf'][0]) >= 0 and   int(args['lf'][0]) <=  int(vl.get(cv2.CAP_PROP_FRAME_COUNT)):
                    look_frame = int(args['lf'][0])
                else:
                    raise Exception('E005')
                if  int(args['lf'][1]) >= 0 and  int(args['lf'][1]) <= int(vl.get(cv2.CAP_PROP_FRAME_COUNT)):
                    dlf = int(args['lf'][1])
                else:
                    raise Exception('E005')
                if look_frame - dlf > 0:
                    l_start_frame = look_frame - dlf
                else:
                    l_start_frame = 0
                if look_frame + dlf < int(vl.get(cv2.CAP_PROP_FRAME_COUNT)):
                    l_end_frame = look_frame + dlf
                else:
                    l_end_frame = int(vl.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            else:
                l_start_frame = 0
                l_end_frame = int(vl.get(cv2.CAP_PROP_FRAME_COUNT)-1)
            str_log = f'The <<look>>  frame range is set {l_start_frame} - {l_end_frame}'#, total frames = {vs.get(cv2.CAP_PROP_FRAME_COUNT)}'
            cf.print_log(str_log, flog=flog, console=debug)

        except Exception as e:
            str_log = f'"{args["lf"]}"\n{str(e)}'
            raise Exception('E005')
            # str_log = f'ERROR!!! (lf).\n{str(e)}'
            # cf.print_log(flog, str_log, console=debug)
            # raise
            # flog.close()
            # exit(222)


        ####################################################################### lr
        try:
            look_roi = None
            if args['lr'] is not None:
                (xl1, yl1, xl2, yl2) = (int(args['lr'][0]), int(args['lr'][1]), int(args['lr'][2]), int(args['lr'][3]))
                if isinstance(args['lr'], (list)) and len(args['lr']) == 4 \
                        and (xl2-xl1) >= 32  and (yl2 - yl1) >= 32:
                    look_roi = (xl1, yl1, xl2, yl2)
                    # str_log = f'The  parameter <<lr>>  has passed: {args["lr"]}'
                    # cf.print_log(flog, str_log, console=debug)
                else:
                    raise Exception('E006')
            else:
                str_log = f'The  parameter <<lr>>: {args["lr"]}. The full frame size is used.'
                cf.print_log(str_log, flog=flog, console=debug)
                (xl1, yl1, xl2, yl2) =  (0, 0, int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                look_roi = (xl1, yl1, xl2, yl2)
            str_log = f'The <<look_roi>>  is set in {look_roi}'
            cf.print_log(str_log, flog=flog, console=debug)
        except Exception as e:
                str_log = f'"{args["lr"]}"\n{str(e)}'
                raise Exception('E006')
            # str_log = f'ERROR!!! (lr).\n{str(e)}'
            # cf.print_log(flog, str_log, console=debug)
            # raise
            # flog.close()
            # exit(111) # Неверно задан параметр lr, используйте x1 y1  x2 y2 - координаты прямоугольной области
            #           # (целые числа, без запятой через пробел)

        ####################################################################### sf
        try:
            if args['sf'] is not None  and  int(args['sf']) >= 0\
                    and int(args['sf']) <= int(vs.get(cv2.CAP_PROP_FRAME_COUNT)):
                src_frame = int(args['sf'])
                str_log = f'The SRC  frame  is set {src_frame}'  # , total frames = {vs.get(cv2.CAP_PROP_FRAME_COUNT)}'
                cf.print_log(str_log, flog=flog, console=debug)
            else:
                raise Exception('E007')
        except Exception as e:
            str_log = f'"{args["sf"]}"\n{str(e)}'
            raise Exception('E007')
            # str_log = f'ERROR!!! (sf). Parameter -sf transferred {args["sf"]} \n{str(e)}'
            # cf.print_log(flog, str_log, console=debug)
            # flog.close()


        ####################################################################### vsf
        try:
            if args['vsf'] is not None:
                if int(args['vsf']) >= 0  and int(args['vsf']) <= int(vs.get(cv2.CAP_PROP_FRAME_COUNT)):
                    vsf = int(args['vsf'])
                else:
                    raise Exception('E008')
                str_log = f'The frames variability of the SRC <<vsf>> is set {vsf}'
                cf.print_log(str_log, flog=flog, console=debug)
                if src_frame - vsf > 0:
                    s_start_frame = src_frame - vsf
                else:
                    s_start_frame = 0
                if src_frame + vsf < int(vs.get(cv2.CAP_PROP_FRAME_COUNT)):
                    s_end_frame = src_frame + vsf
                else:
                    s_end_frame = int(vs.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            else:
                s_start_frame = 0
                s_end_frame = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
            src_fr_range_len = s_end_frame - s_start_frame
            str_log = f'The <<source>>  frame range is set {s_start_frame} - {s_end_frame}'  # , total frames = {vs.get(cv2.CAP_PROP_FRAME_COUNT)}'
            cf.print_log(str_log, flog=flog, console=debug)
        except Exception as e:
            str_log = f'"{args["vsf"]}"\n{str(e)}'
            raise Exception('E008')
            # str_log = f'ERROR!!! (vsf). Parameter -vsf transferred {args["vsf"]} \n{str(e)}'
            # cf.print_log(flog, str_log, console=debug)
            # raise
            # flog.close()
            # exit(222)

        ####################################################################### sr
        try:
            src_roi = None
            if args['sr'] is not None:
                (xs1, ys1, xs2, ys2) = (int(args['sr'][0]), int(args['sr'][1]), int(args['sr'][2]), int(args['sr'][3]))
                if isinstance(args['sr'], (list)) and len(args['sr']) == 4 \
                        and (xl2 - xl1) > (xs2 - xs1) >= 16 and (yl2 - yl1) > (ys2 - ys1) >= 16\
                        and xl2 <= int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)) and yl2 <= int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)):
                    src_roi = (xs1, ys1, xs2, ys2)
                else:
                    raise Exception('E009')
            else:
                str_log = f'The  parameter <<sr>>: {args["sr"]}. The 16X16 src size is used.'
                cf.print_log(str_log, flog=flog, console=debug)
                (xs1, ys1, xs2, ys2) = (xl1 + int((xl2-xl1)/2) - 8, yl1 + int((yl2-yl1)/2) - 8,
                                               xl1 + int((xl2-xl1)/2) + 8, yl1 + int((yl2-yl1)/2) + 8)
                src_roi = (xs1, ys1, xs2, ys2)
            str_log = f'The <<src_roi>>  is set in {src_roi}'
            cf.print_log(str_log, flog=flog, console=debug)
        except Exception as e:
            str_log = f'"{args["sr"]}"\n{str(e)}'
            raise Exception('E009')
            # flog.close()
            # exit(111)  # Неверно задан параметр sf, используйте x1 y1  x2 y2 - координаты прямоугольной области (целые числа,
            #            # без запятой через пробел)

        ####################################################################### vsr
        try:
            if args['vsr'] is not None:
                # if isinstance(args['vsr'], (int)):
                    vsr = int(args['vsr'])
                    str_log = f'The coordinates variability of the SRC <<vsr>> is set {vsr}'
                    cf.print_log(str_log, flog=flog, console=debug)
                    if xs1 - math.ceil(vsr * (xs2-xs1)/100) >= 0:
                        vx1 = - math.ceil(vsr * (xs2-xs1)/100)
                    else:
                        vx1 = - xs1
                    if ys1 - math.ceil(vsr * (ys2 - ys1) / 100) >= 0:
                        vy1 = - math.ceil(vsr * (ys2 - ys1) / 100)
                    else:
                        vy1 = - ys1
                    if xs2 + math.ceil(vsr * (xs2-xs1)/100) <= int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)):
                        vx2 = math.ceil(vsr * (xs2-xs1)/100)
                    else:
                        vx2 = math.ceil(vs.get(cv2.CAP_PROP_FRAME_WIDTH)) - xs2
                    if ys2 + math.ceil(vsr * (ys2 - ys1) / 100) <= int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)):
                        vy2 = math.ceil(vsr * (ys2 - ys1) / 100)
                    else:
                        vy2 = math.ceil(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)) - ys2
                    vrt = (vx1, vy1, vx2, vy2)
                    vrt_len = 4 * vx2 * vy2 + 2 * vy2 + 2 *vx2 + 1
                # else:
                #     raise Exception('E010')
            else:
                str_log = f'The coordinate changer set to (left, top, right, bottom)) {vrt}'
                cf.print_log(str_log, flog=flog, console=debug)
                vrt = (vx1, vy1, vx2, vy2) = (0, 0, 0, 0)
                vrt_len = 4 * vx2 * vy2 + 2 * vy2 + 2 *vx2 + 1
            str_log = f'The coordinate changer set to (left, top, right, bottom)) {vsr}'#, total frames = {vs.get(cv2.CAP_PROP_FRAME_COUNT)}'
            cf.print_log(str_log, flog=flog, console=debug)
        except Exception as e:
            str_log = f'"{args["vsr"]}"\n{str(e)}'
            raise Exception('E010')
            # str_log = f'ERROR!!! (vsr). The parameter was passed as {args["vsr"]} \n{str(e)}'
            # cf.print_log(flog, str_log, console=debug)
            # flog.close()
            # exit(222)

         ####################################################################### socket
        connect = 0
        try:
            sock = socket.socket()
            if isinstance(args['progress'], (int)):
                sock.connect(('localhost', args['progress']))
                connect = 1
                str_log = f'The connection is established'
                cf.print_log(str_log, flog=flog, console=debug)
        except Exception as e:
            str_log = f'ERROR!!! (socket).\n{str(e)}'
            cf.print_log(str_log, flog=flog, console=debug)

        ##################################################################### data_path

        try:
            dpath = cf.set_folder(args['data'], rewrt=1)
            # temp_path = os.getcwd() + '\\' + 'temp'
            str_log = f'Data path: <<{dpath}>>'
            cf.print_log(str_log, flog=flog, console=debug)
        except Exception as e:
            str_log = f'ERROR!!! (data_path) \n{str(e)}'
            cf.print_log(str_log, flog=flog, console=debug)
            flog.close()
            exit(106)  # Невозможно создать каталог для данных
    except Exception as e:
        str_log = f'Error processing parameters. See log...\n{str(e)} {start_errors[str(e)]} {str_log}'
        cf.print_log(str_log, flog=flog, console=debug)
        flog.close()
        exit(-1)  #  ошибка при старте программы

    # MAIN -------------------- MAIN --------------------- MAIN --------------------- MAIN --------------------- MAIN

    try:
        crop_rect = np.array(src_roi)
        vrt2chunk = np.array(vrt)
        chunk_src = crop_rect + vrt2chunk
        chunk_look = np.array((xl1, yl1, xl2, yl2))
        wc, hc = xs2 - xs1, ys2 - ys1
        wl, hl = xl2 - xl1, yl2 - yl1

        range_crops = list(range(0, vrt_len * src_fr_range_len))
        ran_look = list(range(l_start_frame, l_end_frame + 1))

        cutter_start = [k + ran_look[0] for k in range(0, len(ran_look)) if k % 10 == 0]
        cutter_end = cutter_start.copy()
        cutter_end.append(ran_look[-1] + 1)
        cutter_end = cutter_end[-len(cutter_end) + 1:]

        look__X2, look_fX, crop_Y2, crop_fY, lcrops, lready_rdm = [], [], [], [], [], []
        #
        try:

            lcrops = f2cf.fragment_slicer(vs, s_start_frame, s_end_frame, (1,1), (wc,hc), chunk_src, flog=flog,
                                          console=debug)
                # with open(os.path.join(dpath, f'lcrops.pcl'), 'wb') as f:
                #     pickle.dump(lcrops, f)
                # f.close()
            if recalc:
                crop_Y2, crop_fY = f2cf.calc2compare4crop(lcrops, (wl,hl), console=1, flog=flog)
                np.save(os.path.join(dpath, f'crop_fy.npy'),crop_fY)
                np.save(os.path.join(dpath, f'crop_y2.npy'), crop_Y2)
        except Exception as e:
            str_log =  f'ERROR!!! (crop_calc execution error) \n{str(e)}'
            cf.print_log(str_log, flog=flog, console=debug)
            raise

        try:
            if recalc:
                for frs, fre in zip(cutter_start, cutter_end):
                    llook = f2cf.fragment_slicer(vl, frs, fre-1, (1,1), (wl,hl), chunk_look,
                                                  console=1, flog=flog)
                    look_X2, look_fX = f2cf.calc2compare4look(llook, (wc,hc), flog=flog, console=debug)
                    np.save(os.path.join(dpath, f'look_fx-{frs}-{fre - 1}.npy'),look_fX)
                    np.save(os.path.join(dpath, f'look_x2-{frs}-{fre - 1}.npy'), look_X2)
                    # ### look2compare_calc(frs, fre, llook, crop_size, dpath, console=0, flog=None):
                    # variable = Thread(target=look2compare_calc, args=(frs, fre, llook.copy(), (wc, hc), dpath,  debug, flog))  # left = block_rect[0] top = block_rect[1] right = block_rect[2] bottom = block_rect[3]  (100, 220, 260, 320)
                    # variable.setDaemon(True)
                    # variable.start()
                    # while threading.activeCount() > 8:
                    #     print('threading.activeCount', threading.activeCount())
                    #     # print(threading.enumerate())
                    #     time.sleep(1)

        except Exception as e:
            str_log = f'ERROR!!! (look_calc execution error) \n{str(e)}'
            cf.print_log(str_log, flog=flog, console=debug)
            raise

        ## -fin "//Zstorage/td/QA_DEP/PROJECTS/SIS II/Sound/Script_all_formats/video/Видео с регистраторов/CYCLONE DVF 76_ пример видео с автомобильного видеорегистратора.mp4"  -lf 60 90 -lr 360 260 800 600 -sf 70 1 -sr 565 400 640 470 -v 3 -data C:\tttt\1\2 -verbosity DEBUG
        ## -fin "//Zstorage/td/QA_DEP/PROJECTS/SIS II/Sound/Script_all_formats/video/Видео с регистраторов/CYCLONE DVF 76_ пример видео с автомобильного видеорегистратора.mp4"  -lf 150 150 -lr 360 260 900 550 -sf 90 1 -sr 565 400 640 470 -v 2 -data C:\tttt\1\2 -verbosity DEBUG
        ## -fin "//Zstorage/td/QA_DEP/PROJECTS/SIS II/Sound/Script_all_formats/video/Видео с регистраторов/CYCLONE DVF 76_ пример видео с автомобильного видеорегистратора.mp4"  -lf 150 150 -lr 360 260 900 550 -sf 90 1 -sr 570 400 586 476 -v 2 -data C:\tttt\1\2 -verbosity DEBUG
        ## -fin "//Zstorage/td/QA_DEP/PROJECTS/SIS II/Sound/Script_all_formats/video/Видео с регистраторов/CYCLONE DVF 76_ пример видео с автомобильного видеорегистратора.mp4"  -lf 150 150 -lr 900 60 1200 350 -sf 25 1 -sr 1100 140 1116 156 -v 2 -data C:\tttt\1\2 -verbosity DEBUG

        ## -fin C:\test1\test.mp4 -lf 15 15 -lr 363 284 771 682 -sf 15 -vsf 0 -sr 422 942 512 1035 -vsr 0 -data C:\tttt\1\2 -verbosity  DEBUG  -rm 0
        ## -fin C:\test1\test.mp4 -lf 1 29 -lr 2 2 1900 1060 -sf 29 0 -sr 60 60 120 120 -v 3 -data C:\tttt\1\2 -verbosity DEBUG
        ## -fin C:\test1\test.mp4 -lf 1700 50 -lr 1030 360 1345 1010 -sf 25 -vsf 0 -sr 1130 900 1190 960   -data C:\tttt\1\2 -verbosity DEBUG -flin C:\video\fight.mp4 -rm 1
        ## -fin "//Zstorage/td/QA_DEP/PROJECTS/SIS II/Sound/Script_all_formats/video/Видео с регистраторов/CYCLONE MR 90_ видео с передней камеры зеркала-видеорегистратора.mp4"  -lf 25 25 -lr 10 10 1200 430 -sf 25 1 -sr 1000 100 1116 156 -v 1 -data C:\tttt\1\2 -verbosity DEBUG
        ## -fin "//Zstorage/td/QA_DEP/PROJECTS/SIS II/Sound/Script_all_formats/video/Видео с регистраторов/CYCLONE MR 90_ видео с передней камеры зеркала-видеорегистратора.mp4"  -lf 20 20 -lr 10 10 1200 540 -sf 20 1 -sr 1000 100 1116 156 -v 1 -data C:\tttt\1\2 -verbosity DEBUG
        ## -fin "//Zstorage/td/QA_DEP/PROJECTS/SIS II/Sound/Script_all_formats/video/Видео с регистраторов/CYCLONE DVF 76_ пример видео с автомобильного видеорегистратора.mp4"  -lf 70 56 -lr 10 10 600 400 -sf 70 1 -sr 190 80 290 170 -v 1 -data C:\tttt\1\2 -verbosity DEBUG
        ## -fin "//Zstorage/td/QA_DEP/PROJECTS/SIS II/Sound/Script_all_formats/video/Видео с регистраторов /CYCLONE DVF 76_ пример видео с автомобильного видеорегистратора.mp4"  -lf 64 59 -lr 500 60 1200 350 -sf 22 1 -sr 1000 140 1116 156 -v 1 -data C:\tttt\1\3 -verbosity DEBUG
        # -fin "//Zstorage/td/QA_DEP/PROJECTS/SIS II/Sound/Script_all_formats/video/Видео с регистраторов/CYCLONE DVF 76_ пример видео с автомобильного видеорегистратора.mp4"  -lf 150 150 -lr 500 60 1200 350 -sf 60 1 -sr 1000 140 1116 156 -v 3 -data C:\tttt\1\3 -verbosity DEBUG
        # t = time.time()
        ## -fin "//Zstorage/td/QA_DEP/PROJECTS/SIS II/Sound/Script_all_formats/video/Видео с регистраторов/CYCLONE DVF 76_ пример видео с автомобильного видеорегистратора.mp4"  -lf 300 300 -lr 360 260 800 600 -sf 70 0 -sr 525 450 600 505 -v 0 -data C:\tttt\1\5 -verbosity DEBUG -rm 0
        ## -fin "C:\video\fight.mp4"  -lf 150 150 -lr 1 1 1200 720 -sf 25 0 -sr 1000 100 1016 156 -v 0 -data C:\tttt\1\2 -verbosity DEBUG -rm 0
        ## Где стояла тетка -fin "C:\video\fight.mp4"  -lf 200 329 -lr 448 292 700 720 -sf 100 0 -sr 559 565 660 660 -v 0 -data C:\tttt\1\f_max_fl64 -verbosity DEBUG -rm 1
        ## -fin "C:\video\fight.mp4"  -lf 62 59 -lr 1 1 1200 720 -sf 50 0 -sr 1000 100 1016 156  -v 0 -data C:\tttt\1\fl64 -verbosity DEBUG -rm 0


        # new

        ## -fin C:\test1\test.mp4 -lf 1700 150 -lr 1030 360 1345 1010 -sf 27 -vsf 0 -sr 1130 970 1220 1000 -vsr 0  -data C:\tttt\1\2 -verbosity DEBUG -flin C:\video\fight.mp4 -rm 0
        ## //Zstorage/td/QA_DEP/PROJECTS/SIS II/Sound/Script_all_formats/video/Видео с регистраторов/CYCLONE DVF 76_ пример видео с автомобильного видеорегистратора.mp4"
        ## -fin "//Zstorage/td/QA_DEP/PROJECTS/SIS II/Sound/Script_all_formats/video/Видео с регистраторов/CYCLONE DVF 76_ пример видео с автомобильного видеорегистратора.mp4"  -lf 60 90 -lr 360 260 800 600 -sf 70 -vsf 1 -sr 565 400 640 470 -vsr 1 -data C:\tttt\1\2 -verbosity DEBUG -rm 1
        try:
            if recalc:
                crop_Y2 = np.load(os.path.join(dpath, f'crop_y2.npy'))
                crop_fY = np.load(os.path.join(dpath, f'crop_fy.npy'))
        except Exception as e:
            str_log =  f'ERROR!!! (crop reading error) \n{str(e)}'
            cf.print_log(str_log, flog=flog, console=debug)

            raise

        try:
            if recalc:
                for frs, fre in zip(cutter_start, cutter_end):
                    str_log = f'Start COS_dist_calc for frames:  {frs}-{fre - 1}'
                    cf.print_log(str_log, flog=flog, console=debug)
                    try:
                        look_X2 = np.load(os.path.join(dpath, f'look_x2-{frs}-{fre - 1}.npy'))
                        look_fX = np.load(os.path.join(dpath, f'look_fx-{frs}-{fre - 1}.npy'))
                    except Exception as e:
                        str_log = f'ERROR!!! (look reading error) \n{str(e)}'
                        cf.print_log(str_log, flog=flog, console=debug)
                        raise
                    variable = Thread(target=COS_dist_calc, args=(frs, fre, look_X2.copy(), look_fX.copy(), crop_Y2, crop_fY,
                                                                  dpath, lready_rdm, flog, debug))
                    variable.setDaemon(True)
                    variable.start()
                    while threading.activeCount() > 6:
                        #print('Number of active threads', threading.activeCount())
                        #print(threading.enumerate())
                        time.sleep(1)
        except Exception as e:
            str_log = f'ERROR!!! (calc_rdm {frs}-{fre - 1}) \n{str(e)}'
            cf.print_log(str_log, flog=flog, console=debug)
            raise

        lll = []
        cv2.namedWindow(MAIN_WINDOW_NAME)

        # Result WINDOW
        lim_bb = 1
        cv2.createTrackbar(TRACK_BB_NAME, MAIN_WINDOW_NAME, 1, 100, set_lim_bb)
        cv2.setTrackbarPos(TRACK_BB_NAME, MAIN_WINDOW_NAME, lim_bb)  #

        lim_rdm = 20
        cv2.createTrackbar(TRACK_COS_NAME, MAIN_WINDOW_NAME, 0, 100, set_lim_rdm)
        cv2.setTrackbarPos(TRACK_COS_NAME, MAIN_WINDOW_NAME, lim_rdm)

        rdm_level = np.exp(2.3025850929940456840179914546844 + (lim_rdm / 10 - 1))
        rdm_level = rdm_level / (1 + rdm_level)

        lim_ssq2pix = 0
        cv2.createTrackbar(TRACK_P2P_NAME, MAIN_WINDOW_NAME, 0, 100, set_lim_ssq2pix)
        cv2.setTrackbarPos(TRACK_P2P_NAME, MAIN_WINDOW_NAME, lim_ssq2pix)  #

        lim_h = 4
        cv2.createTrackbar(TRACK_HAM_NAME, MAIN_WINDOW_NAME, 0, 100, set_lim_h)
        cv2.setTrackbarPos(TRACK_HAM_NAME, MAIN_WINDOW_NAME, lim_h)  #

        # try:
        #     with open(os.path.join(dpath, f'lcrops.pcl'), 'r') as f:
        #         lcrops = pickle.load(f, encoding='bytes')
        #     f.close()
        # except Exception as e:
        #     cf.print_log(flog, f'ERROR!!! (load lcrops {os.path.join(dpath, "lcrops.pcl")})\n{str(e)}', console=debug)
        #     raise

        for r, rr in zip(cutter_start, cutter_end):
            if recalc:
                while (r,rr) not in lready_rdm:
                    time.sleep(1)
                    print('.', end='')

            t = time.time()
            # lim = 0.96#lim_rdm / 100 #0.97
            rdm = load_rdm(r, rr, rdm_level,  dpath, flog=flog, console=debug) #np.load(f'{r}-{rr - 1}.npy')/1000000000
            # print('load', time.time() - t)
            # t = time.time()
            # # rdm = rdm[:, 0:1, :,:]
            # # rdm[rdm < lim] = 0
            str_log = f'\nREADY {r} --- {rr-1} in time {time.time()-t}'
            cf.print_log(str_log, flog=flog, console=debug)
            for dd, num_ in zip(rdm, list(range(0, rdm.shape[0]))):
                rdm_level = np.exp(2.3025850929940456840179914546844 + (lim_rdm / 10 - 1))
                rdm_level = rdm_level / (1 + rdm_level)
                # time.sleep(10)
                clone_num = int(dd.shape[0] / 2)
                num_look_fr = num_ + r
                ddd = None
                cloneXY = None
                li_temp_to_draw = []
                num_fr_src = 0
                vl.set(cv2.CAP_PROP_POS_FRAMES, num_look_fr)
                ret, look_img = vl.read()
                chunk_look_img = look_img[yl1:yl2, xl1:xl2].copy()
                tempT = np.sum(dd, axis=(3)) #/ dd.shape[3]

                # eee = tempT.copy()/3
                # eee[eee < 0.975] = 0
                # imgsob = (eee[clone_num] * 255).astype(np.uint8)  # func_sob((tempT[clone_num]*255).astype(np.float32)).astype(np.uint8)
                # cv2.imshow('ttttt', imgsob)

                # tempT[tempT < dd.shape[3]*lim] = 0
                N = math.ceil(np.sum(tempT[tempT > dd.shape[3]* rdm_level]) / dd.shape[3])
                str_log = f'  Frame :{num_look_fr:-6} || N:{round(N, 2):9.2f} || max(tempT): {np.max(tempT):1.5f} ||' \
                          f'  hamD {lim_h:-4} || PPxD {lim_ssq2pix:1.5f} || cosD {round(rdm_level,9):1.9}'
                cf.print_log(str_log, flog=flog, console=debug)
                # print(f' Num_fr = {num_look_fr}, N = {N}, max(tempT) = {np.max(tempT)}')
                # if num_look_fr == 1706:
                #     print(1)
                if N > rdm_level:
                    n, c = 0, 0
                    while n in range(0, min(5, math.floor(N))+1):  # look_img.shape[0]):
                        # print(dd.shape[0], math.ceil(N))
                        n += 1
                        mx = np.argmax(tempT, axis=None)
                        cloneXY = np.unravel_index(mx, tempT.shape)
                        tcos = tempT[cloneXY] / dd.shape[3] # , cloneXY[2]]
                        (clone_num, ylc1, xlc1) = cloneXY
                        (xlc1, ylc1, xlc2, ylc2) = (xlc1, ylc1, xlc1 + wc, ylc1 + hc) # Clone in LOOK
                        (xc1, yc1, _) = lcrops[clone_num]
                        (xc1, yc1, xc2, yc2) = (xc1, yc1, xc1 + wc, yc1 + hc)  # Clone
                        num_fr_src = s_start_frame + math.floor(clone_num/vrt_len)
                        # if num_look_fr == 29:
                        #     print(1)
                        if tempT[cloneXY] == 0:  # , cloneXY[2]] == 0:
                            break
                        # if xlc2 > chunk_look_img.shape[1] or  ylc2 > chunk_look_img.shape[0]:
                        #     break
                        else:
                            # d1[cloneXY[1]:cloneXY[1]+hc, cloneXY[0]:cloneXY[0]+wc] = mmm
                            # print('n', n)
                            tempT[clone_num, ylc1:ylc2, xlc1:xlc2] = 0
                            vs.set(cv2.CAP_PROP_POS_FRAMES, num_fr_src)
                            ret, src_img = vs.read()
                            _crop_ = src_img[yc1:yc2, xc1:xc2]
                            _crop_in_look_ = look_img[ylc1 + yl1:ylc2+ yl1, xlc1+xl1:xlc2+xl1]
                            test1 = np.concatenate((_crop_, _crop_in_look_))
                            # cv2.imshow('_crop_', _crop_)
                            # cv2.imshow('_crop_in_look_', _crop_in_look_)
                            # cv2.waitKey(1)
                            # print(_crop_.shape, _crop_in_look_.shape)
                            # print(imagehash.phash(Image.fromarray(_lok.astype(np.uint8))) - imagehash.phash(Image.fromarray(_src.astype(np.uint8))))
                            h = f2cf.hamming_dist2img(_crop_in_look_, _crop_)
                            s2pix = f2cf.dif2pix(_crop_in_look_, _crop_)
                            cos = tcos
                            s2pix_sq = s2pix / (wc * wl)
                            if h <= lim_h/1 or s2pix_sq <= lim_ssq2pix/1:
                                li_temp_to_draw.append((xlc1, ylc1, xlc2, ylc2))
                                p_h = math.exp(-(h-8))/(1+math.exp(-(h-8)))
                                p_s2pix = math.exp(-(s2pix_sq - 12)) / (1 + math.exp(-(s2pix_sq - 12)))
                                p_res = max((1 - s2pix), cos * p_h * p_s2pix)
                                lll.append((num_look_fr, num_fr_src, xlc1+xl1, ylc1+yl1,
                                    xc1, yc1, wc, hc, cos, h, s2pix, s2pix_sq, p_res,
                                    math.floor(N), p_h, p_s2pix))#-0.001*s2pix_sq)/15
                                cv2.imwrite(os.path.join(dpath, f'{num_look_fr}-{num_fr_src}-{n}-{len(lll)-1}-{p_res}.png'), test1)
                                str_log = f'    Cloned fragment detected in frame # {num_look_fr} from cropFR/var: {num_fr_src}/{n}-{clone_num}: \n' \
                                          f'      -- p_res= {p_res}, p_h= {p_h}, p_s2p= {p_s2pix}, N= {N}, \n' \
                                          f'      -- ham_dist = {h}, s2pq = {round(s2pix_sq,2)}, s2pix = {s2pix}, cos_dist = {cos},\n' \
                                          f'      -- look_coor ({xlc1 + xl1}, {ylc1 + yl1}),\n' \
                                          f'      -- crop_coor ({xc1}, {yc1})'

                                cf.print_log(str_log, flog=flog, console=debug)
                    # img1 = img[yc:yc+hc, xc:xc+wc]

                    # dd[dd > 1] = 0  #
                    # dd[dd < 0.8] = 0.8
                    # dd = (255+1000*np.log(dd)).astype(np.uint8)

                dd = dd[clone_num]
                dd[dd > 1] = 1
                # th1 = 10 * np.exp(10 * (lim_rdm / 100 - 0.1))
                dd[dd < rdm_level] = 0
                # dd[dd < 1-1/(lim_rdm+15)] = 0
                ttt = np.exp(10 * (dd - 0.95))
                dd =  (255 * ttt/(1+ttt)).astype(np.uint8)
                # Сдвиг половинка Crop
                xt1, yt1 = int(wc / 2), int(hc / 2)
                t1 = np.zeros(dd.shape).astype(np.uint8) + (np.average(dd, axis=(0, 1)).astype(np.uint8))
                t2 = dd[0: dd.shape[0] - yt1, 0: dd.shape[1] - xt1]
                t1[yt1:, xt1:] = t2
                dd = t1.copy()
                dd = cv2.GaussianBlur(dd, (9, 9), 0)

                for xlc1, ylc1, xlc2, ylc2 in li_temp_to_draw:
                    dd[ylc1:ylc2, xlc1:xlc2] = 150
                    img_ = dd[ylc1:ylc2, xlc1:xlc2].copy()
                    g = f2cf.gaus2d(img_.shape[1], img_.shape[0], mx=0, my=0, sx=1, sy=1)
                    g = np.stack((g, g, g))
                    g = np.transpose(g, (1, 2, 0))
                    img_ = 5 * g * img_
                    dd[ylc1:ylc2, xlc1:xlc2] = img_.astype(np.uint8)
                    # dd = cv2.rectangle(dd, (ll[1], ll[0]), (ll[1] + wc, ll[0] + hc), (0, 0, 255), 2)

                if dd.shape != chunk_look_img.shape:
                    temp = np.zeros(chunk_look_img.shape).astype(np.uint8)
                    temp[0:dd.shape[0], 0:dd.shape[1], :] = dd
                    dd = temp.copy()

                ddd = cv2.addWeighted(chunk_look_img, 1-lim_bb/100, dd, 1 + lim_bb/100, 0)

                for xlc1, ylc1, xlc2, ylc2 in li_temp_to_draw:
                    ddd = cv2.rectangle(ddd, (xlc1, ylc1), (xlc2, ylc2), (0, 0, 255), 2)

                if ddd is not None:
                    look_img[yl1:yl2, xl1:xl2] = ddd
                # вставляем  в look со сдвигом в половину кропа
                # img[yl1+int(hc/2):yl1+int(hc/2)+te.shape[0],xl1+int(wc/2):xl1+int(wc/2)+te.shape[1]] = ddd[int(hc/2):int(hc/2)+te.shape[0] , int(wc/2):int(wc/2)+te.shape[1]]

                # img[yl1 + int(wc / 2):yl1 + int(wc / 2) + ddd.shape[0],xl1 + int(hc / 2):xl1 + int(hc / 2) + ddd.shape[1]] = ddd

                dy = max(45, yl1 - 65)
                color = (50, 50, 255) # random.randint(200,255))
                look_img = cv2.rectangle(look_img, (xl1, yl1), (xl2, yl2), (255, 255, 255), 2)
                if cloneXY is not None:
                    look_img = cv2.putText(look_img, f'Frame {num_look_fr} x:{cloneXY[2]} y:{cloneXY[1]}',
                                    (xl1-3, dy), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 5)
                    look_img = cv2.putText(look_img, f'cosD {round(rdm_level,9)},  PPxD {lim_ssq2pix}, hamD {lim_h}',
                                    (xl1-3, dy + 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 5)
                    look_img = cv2.putText(look_img, f'Frame {num_look_fr} x:{cloneXY[2]} y:{cloneXY[1]}',
                                    (xl1+3, dy), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 5)
                    look_img = cv2.putText(look_img, f'cosD {round(rdm_level,9)},  PPxD {lim_ssq2pix}, hamD {lim_h}',
                                    (xl1+3, dy + 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 5)
                    look_img = cv2.putText(look_img, f'Frame {num_look_fr} x:{cloneXY[2]} y:{cloneXY[1]}',
                                    (xl1, dy), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
                    look_img = cv2.putText(look_img, f'cosD {round(rdm_level,9)},  PPxD {lim_ssq2pix}, hamD {lim_h}',
                                    (xl1, dy + 35), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
                else:
                    look_img = cv2.putText(look_img, f'Frame {num_look_fr} x:- y:- ',
                                    (xl1-3, dy), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 5)
                    look_img = cv2.putText(look_img, f'cosD {round(rdm_level,9)},  PPxD {lim_ssq2pix}, hamD {lim_h}',
                                    (xl1-3, dy + 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 5)
                    look_img = cv2.putText(look_img, f'Frame {num_look_fr} x:- y:- ',
                                    (xl1+3, dy), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 5)
                    look_img = cv2.putText(look_img, f'cosD {round(rdm_level,9)},  PPxD {lim_ssq2pix}, hamD {lim_h}',
                                    (xl1+3, dy + 35), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 5)
                    look_img = cv2.putText(look_img, f'Frame {num_look_fr} x:- y:- ',
                                    (xl1, dy), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
                    look_img = cv2.putText(look_img, f'cosD {round(rdm_level,9)},  PPxD {lim_ssq2pix}, hamD {lim_h}',
                                    (xl1, dy + 35), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)


                # img[yl1:yl2, xl1:xl2] = ddd
                cv2.imshow(MAIN_WINDOW_NAME, f2cf.resize_img(look_img))# f2cf.resize_img(
                cv2.waitKey(1)

        print('len llxy', len(lll))
        if len(lll) > 0:
            d_arr = np.array(lll)
            np.save(os.path.join(dpath, 'res_np' + '.npy'), d_arr)
            df = pd.DataFrame(d_arr)
            # _fr, num_fr_src, xlc1 + xl1, ylc1 + yl1, xlc2 + xl1, ylc2 + yl1,
            # xc1, yc1, xc2, yc2, cos, h, s2pix, s2pix_sq, max((1 - s2pix),
            #                                                  cos * math.exp(-h * 0.01) * math.exp(
            #                                                      (-0.00006 * s2pix_sq)))))  # -0.001*
            df.rename(columns = {0:'frl', 1:'frc', 2:'xl', 3:'yl', 4:'xs', 5:'ys', 6:'wc', 7:'hc', 8:'cos', 9:'h',
                                 10:'s2pix', 11:'s2pix_sq', 12:'p', 13:'N', 14:'p_h', 15:'p_s2pix'}, inplace=1)
            df['frl'] = df['frl'].astype('Int32')
            df['frc'] = df['frc'].astype('Int32')
            df['xl'] = df['xl'].astype('Int32')
            df['yl'] = df['yl'].astype('Int32')
            df['xs'] = df['xs'].astype('Int32')
            df['ys'] = df['ys'].astype('Int32')
            df['wc'] = df['wc'].astype('Int32')
            df['hc'] = df['hc'].astype('Int32')
            df['cos'] = df['cos'].astype('float32')
            df['h'] = df['h'].astype('Int32')
            df['s2pix'] = df['s2pix'].astype('Int32')
            df['s2pix_sq'] = df['s2pix_sq'].astype('float32')
            df['p'] = df['p'].astype('float32')
            df['N'] = df['N'].astype('Int32')
            df['p_h'] = df['p_h'].astype('float32')
            df['p_s2pix'] = df['p_s2pix'].astype('float32')

            # df = df.sort_values(by=["p"], ascending=[False])
            # df = df.sort_values(by = ["p", "s2pix_sq", "N"], ascending=[True, False, True])
            #
            df.to_csv(os.path.join(dpath, 'res_pandas' + '.csv'), sep=',', index=False)

        flog.close()
    except Exception as e:
        str_log = f'ERROR!!! (main) \n {str(e)}'
        cf.print_log(str_log, flog=flog, console=debug)
        flog.close()
        raise
        exit(222)  # Внутренняя ошибка

# rax = 453 + k
# ray = 296 + k
# cv2.imwrite('src_block1-' + str(i) + '-' + str(j) + '.png', _frsrc_[ray:ray + 100, rax:rax + 100])
# cv2.imwrite('look_block1-' + str(i) + '-' + str(j) + '.png', look[941:941 + 100, 420:420 + 100])
# rrrrr = np.abs(
#     look.astype(np.int32)[941:941 + 100, 420:420 + 100] - _frsrc_.astype(np.int32)[ray:ray + 100, rax:rax + 100])
# cv2.imwrite('del_block1-' + str(j) + '.png', 255 - rrrrr)
# print('rrrrr', np.sum(rrrrr))
# rax = 545 + k
# ray = 564 + k
# #     k +=7
# #     t[441:441+40, 1693:1693+40] = te[ray:ray+40, rax:rax+40]
# cv2.imwrite('src_block2-' + str(i) + '-' + str(j) + '.png', _frsrc_[ray:ray + 40, rax:rax + 40])
# cv2.imwrite('look_block2-' + str(i) + '-' + str(j) + '.png', look[441:441 + 40, 1693:1693 + 40])

# src_b2 = pipe2img(_frsrc_[ray:ray + 40, rax:rax + 40], 0)
                # look_b2 = pipe2img(look[441:441 + 40, 1693:1693 + 40], 0)
                # cv2.imwrite('src_block2-'  + str(i)+'-'+ str(j) +'.png', src_b2)
                # cv2.imwrite('look_block2-' +  str(i)+'-'+ str(j) + '.png',look_b2)
                # rrrr2 = np.abs(look_b2.astype(np.int32) - src_b2.astype(np.int32))
                # cv2.imwrite('del_block2-' + str(j) + '.png', 255 - rrrr2)
                # print('rrrr2', np.sum(rrrr2))
