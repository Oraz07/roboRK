#!/usr/bin/env python
import subprocess as sub
import time
from os import remove, path, listdir
import cv2
import json
import numpy as np
import pytesseract
from PIL import Image
from math import cos, sin, pow, e, sqrt
from random import randint
from scipy.spatial.distance import cosine

DEVICE_PATH = '/sdcard/'
RES_SCALE = 0.4
CROP_X = 296
CROP_Y = 155
CROP_X_OFFSET = 26
CROP_Y_OFFSET = 15

ELLIPSE_EXTENSION = 1.2
LATEST_CAMP = 0
X_TROOP_BUTTON = 25
FIRST_TROOP_BUTTON = 950
SECOND_TROOP_BUTTON = 807
SWIPE_DELAY = 500
STEP_SWITCH_STATS = 35
STATS_X = 150
STATS_Y = 145
WHERE_I_AM = ""

class Troop(object):
    name = ""
    level = 0
    count = 0

    def __init__(self, name, level, count):
        self.name = name
        self.level = level
        self.count = count

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=False, indent=4)


def launch_cmd(cmd):
    output = sub.check_output(cmd.split())
    return output.decode('UTF-8', 'ignore').split()

def define_troops():
    ret_val = []
    with open('troops.json','r') as f:
        troops = json.loads(f.read())
    for t in troops['troops']:
        ret_val.append(Troop(t['name'], t['level'], t['count']))
    return ret_val

def progressive_nameFile( dir ):
    max = app = 0
    for filename in listdir(dir):
        app = int(filename[4:-4])
        if app > max:
            max = app
    return 'camp' + str(max+1) + '.png', max+1

def take_screenshot():
    # take the screenshot and return the name of the file.
    localtime = time.localtime()
    nameFile = time.strftime("%Y%m%d%H%M%S", localtime) + '.png'
    launch_cmd('adb shell screencap -p ' + DEVICE_PATH + nameFile)
    launch_cmd('adb pull ' + DEVICE_PATH + nameFile)
    launch_cmd('adb shell rm ' + DEVICE_PATH + nameFile)
    # resize the image
    cv2.imwrite("highres.png" ,img)
    img = cv2.imread(nameFile)
    img = cv2.resize(img, (0,0), fx=RES_SCALE, fy=RES_SCALE)
    cv2.imwrite(nameFile ,img)
    return nameFile

def take_camp_screenshot():
    global LATEST_CAMP
    # village pinch out and repositioning
    # launch_cmd("adb shell sh %ssendevent_input.sh" % DEVICE_PATH)
    time.sleep(2)
    pinch_out()
    launch_cmd("adb shell input swipe 625 400 125 100 1000")
    # take the camp screenshot and return the name of the file.
    if not LATEST_CAMP:
        nameFile, LATEST_CAMP = progressive_nameFile('camp_screen/')
    else:
        nameFile = 'camp' + str(LATEST_CAMP + 1) + '.png'
        LATEST_CAMP += 1
    launch_cmd('adb shell screencap -p ' + DEVICE_PATH + nameFile)
    launch_cmd('adb pull ' + DEVICE_PATH + nameFile + " camp_screen")
    launch_cmd('adb shell rm ' + DEVICE_PATH + nameFile)
    # resize the image
    img = cv2.imread('camp_screen/' + nameFile)
    img = cv2.resize(img, (0,0), fx=RES_SCALE, fy=RES_SCALE)
    cv2.imwrite('camp_screen/' + nameFile ,img)
    return 'camp_screen/' + nameFile

def locate_sub_image(template_p, screenshot):
    # prende come argomento il path della foto di template e restituisce un vettore di posizioni
    # trovate nello screen corrente
    points = []
    to_remove = False
    if not screenshot:
        screenshot = take_screenshot()
        to_remove = True

    img_rgb = cv2.imread(screenshot)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_p, 0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    if to_remove:
        remove(screenshot)
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        points.append(pt)
    # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    # cv2.imwrite('templateMatch.png',img_rgb)
    return points

def locate_stronghold(camp):
    # prende come argomento il path della foto di template e restituisce un vettore di posizioni
    # trovate nello screen corrente
    points = []
    lev = 1;
    found = False;
    img_gray = cv2.cvtColor(camp, cv2.COLOR_BGR2GRAY)
    while not found and lev < 10:
        print ('templates/stronghold-lev%d.png' % lev)
        template = cv2.imread('templates/stronghold-lev%d.png' % lev, 0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.85
        loc = np.unravel_index(res.argmax(), res.shape) #np.where( res >= threshold)
        if res[loc[0], loc[1]] > threshold :
            return [loc], lev
        else:
            lev += 1
    return [(0,0)], 0

def tap_on_template_found(template_p, screen):
    # invia un tap sulla zona dove viene trovata la prima corrispondenza della
    # ricerca parziale sull'attuale schermo mostrato
    points = locate_sub_image(template_p, screen)
    if points:
        print ("found " +  template_p + " in position " + str(points[0][0]) + " " + str(points[0][1]))
        launch_cmd("adb shell input tap " + str((points[0][0] / RES_SCALE) + 3) + " " + str((points[0][1] / RES_SCALE) + 3))
        return 1
    else:
        print ("no match found for " + template_p)
        return 0

def perc_recognition(stats_screen):
    # recognize the percentage in stats_screen
    to_remove = False
    if not stats_screen:
        stats_screen = take_screenshot()
        to_remove = True
    img = cv2.imread(stats_screen, 0)
    start_x = locate_sub_image('templates/perc.png', stats_screen)[0][0]
    img = img[CROP_Y:CROP_Y+CROP_Y_OFFSET, start_x-CROP_X_OFFSET:start_x]
    ret, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY) # apply threshold
    cv2.imwrite('temp.png', img)
    ret_val = pytesseract.image_to_string(img, config='outputbase digits')
    # remove('temp.png')
    if to_remove:
        remove(stats_screen)
    return int(ret_val)

def take_percs(n):
    #starting from the battle history view, return an array with the last n perc
    y = STATS_Y
    ret = []
    for i in range(n):
        app = perc_recognition(0)
        print(app)
        ret.append(app)
        y += STEP_SWITCH_STATS
        launch_cmd("adb shell input tap %d %d" % (STATS_X / RES_SCALE, y / RES_SCALE))
        time.sleep(1)
    return ret[::-1]

def locate_village():
    # carico le immagini
    print ('load background...')
    bg = cv2.imread('background.png')
    print ('taking screenshot...')
    camp_file_name = take_camp_screenshot()
    camp = cv2.imread(camp_file_name)

    print ('searching for stronghold...')
    points, lev = locate_stronghold(camp)
    stronghold_x = points[0][1] + 10
    stronghold_y = points[0][0] + 10
    print ("Stronghold level " + str(lev) + " in position " + str(stronghold_x) + " " + str(stronghold_y))

    print ('calculate the contour...')
    diff = cv2.subtract(bg, camp)
    # cv2.imwrite('video/diff.png', diff)

    # converto il risultato in scala di grigi
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    ret, diff_gray = cv2.threshold(diff_gray, 47, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('video/diff_gray.png', diff_gray)

    # dilatazione
    kernel = np.ones((3,3),np.uint8)
    dilated = cv2.dilate(diff_gray, kernel, iterations=8)
    # cv2.imwrite('video/dilated.png', dilated)
    # dilated_match = cv2.imread('video/dilated.png')

    # delimito i contorni
    with_cont, contours, _ = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))

    c_areas = [cv2.contourArea(x) for x in contours]

    zipped = list(zip(c_areas, contours))
    cont = max(zipped, key=lambda item:item[0])[1]
    cv2.drawContours(camp, [cont], -1, (0,255,0), 3)
    # cv2.drawContours(dilated_match, [cont], -1, (0,255,0), 3)
    # cv2.circle(dilated_match, (stronghold_x, stronghold_y), 15, (5,205,232), 2)

    if stronghold_x == stronghold_y != 10:
        while cv2.pointPolygonTest(cont, (stronghold_x,stronghold_y), False) < 0 :
            # fino a quando lo stronghold non sta dentro al contorno corrente
            zipped.remove(max(zipped, key=lambda item:item[0]))
            cont = max(zipped, key=lambda item:item[0])[1]

    #disegno ellisse
    print ('Drawing ellipse over the contour...')
    ellipse = cv2.fitEllipse(cont)
    print (ellipse)
    (x_center, y_center), (x_ax, y_ax), angle = ellipse
    x_ax = x_ax * ELLIPSE_EXTENSION
    y_ax = y_ax * ELLIPSE_EXTENSION
    if x_ax < y_ax:
        app = x_ax
        x_ax = y_ax
        y_ax = app
        angle += 90
    ellipse = (x_center, y_center), (x_ax, y_ax), angle
    # ellisse ((coordinate centro), (lunghezza degli assi), inclinazione in gradi)
    cv2.ellipse(camp, ((x_center, y_center),(x_ax, y_ax), angle),(0,0,255),2)
    # cv2.ellipse(dilated_match, ((x_center, y_center),(x_ax, y_ax), angle),(0,0,255),2)

    # cv2.imwrite('video/ellipse.png', camp)
    # cv2.imwrite('video/ellipse.png', dilated_match)


    return ellipse, cont, (int(stronghold_x), int(stronghold_y), int(lev)), camp_file_name

def len_radius(a, b, theta):
    theta = theta * 3.14 / 180.0
    r = (a*b / (sqrt(pow(b*cos(theta),2) + pow(a*sin(theta),2))))/2
    print ("Radius length: " + str(r))
    return r

def findRadius(ellipse, angle):
    (x_center, y_center), (x_ax, y_ax), ellipse_angle = ellipse
    length = len_radius(x_ax, y_ax, angle)
    point_x = x_center + length * cos((angle + ellipse_angle) * 3.14 / 180.0)
    point_y = y_center + length * sin((angle + ellipse_angle) * 3.14 / 180.0)

    return (x_center, y_center), (point_x, point_y)

def load_village():
    go_on = True
    while go_on:
        screen = take_screenshot()
        if len(locate_sub_image("templates/is_in_battle.png", screen)) + len(locate_sub_image("templates/is_in_battle_2.png", screen)):
            go_on = False
        remove(screen)
        time.sleep(2)
    return 1


def launch_troops(ellipse, contour, stronghold_x, stronghold_y, camp_file_name, data):
    cont = 0
    troops = define_troops()
    camp = cv2.imread(camp_file_name)
    (x_center, y_center), (x_ax, y_ax), ellipse_angle = ellipse
    cv2.ellipse(camp, ((x_center, y_center),(x_ax, y_ax), ellipse_angle),(0,0,255),2)
    cv2.drawContours(camp, [contour], -1, (0,255,0), 3)
    cv2.circle(camp, (stronghold_x, stronghold_y), 12, (5,205,232), 2)

    dragon = c = 0
    for t in troops:
        if t.name == "dragon":
            dragon = c + 1
        c+=1
    print ("dragon in position %d of %d troops" % (dragon, len(troops)))

    while cont < len(troops):

        data["troop%s_name" % str(cont+1)] = troops[cont].name
        data["troop%s_level" % str(cont+1)] = troops[cont].level
        data["troop%s_units" % str(cont+1)] = troops[cont].count

        # calc of the right point in function of the random angle
        angle = randint(0, 360)
        print ('\nGenerated angle %d: %d' % (cont+1,angle))
        (p1_x, p1_y),(p2_x, p2_y) = findRadius(ellipse, angle)

        if cv2.pointPolygonTest(contour, (p2_x, p2_y), False) > -1 :
            # -1 outside, 0 in contour, 1 inside
            print ("--> inside the contour, generate another angle!\n")

            cv2.line(camp, (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)),(100,200,0),2)
            continue

        data["troop%s_x" % str(cont+1)] = p2_x
        data["troop%s_y" % str(cont+1)] = p2_y
        data["troop%s_angle" % str(cont+1)] = angle
        cv2.line(camp, (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)),(0,0,255),2)
        cv2.circle(camp, (int(p1_x), int(p1_y)), 2, (255,0,0), 2)
        cv2.circle(camp, (int(p2_x), int(p2_y)), 2, (255,0,0), 2)

        # gesture swipe
        if dragon:
            if cont == dragon-1:   # the dragon button has fixed position in the screen
                launch_cmd("adb shell input swipe %d %d %d %d %d" % (X_TROOP_BUTTON, FIRST_TROOP_BUTTON, p2_x / RES_SCALE, p2_y / RES_SCALE, SWIPE_DELAY))
            else:
                launch_cmd("adb shell input swipe %d %d %d %d %d" % (X_TROOP_BUTTON, SECOND_TROOP_BUTTON, p2_x / RES_SCALE, p2_y / RES_SCALE, SWIPE_DELAY))
        else: # no dragon in the troops, use only FIRST_TROOP_BUTTON
            launch_cmd("adb shell input swipe %d %d %d %d %d" % (X_TROOP_BUTTON, FIRST_TROOP_BUTTON, p2_x / RES_SCALE, p2_y / RES_SCALE, SWIPE_DELAY))

        cont += 1
    cv2.imwrite('res.png', camp)

def cosine_sim(s_name):
    sample = np.ravel(cv2.imread(s_name).astype(float))
    screen = take_screenshot()
    to_test = np.ravel(cv2.imread(screen).astype(float))

    cs = cosine(sample, to_test)
    remove(screen)
    print ("Cosine similarity: ", cs)
    if cs < 0.08:
        return True
    else:
        return False

def end_battle():
    sample_w = np.ravel(cv2.imread("templates/win_battle.png").astype(float))
    sample_w2 = np.ravel(cv2.imread("templates/win_battle_2.png").astype(float))
    sample_w3 = np.ravel(cv2.imread("templates/win_battle_3.png").astype(float))
    sample_w4 = np.ravel(cv2.imread("templates/win_battle_4.png").astype(float))
    sample_l = np.ravel(cv2.imread("templates/lose_battle.png").astype(float))
    screen = take_screenshot()
    to_test = np.ravel(cv2.imread(screen).astype(float))

    if cosine(sample_w, to_test) < 0.085 or cosine(sample_w2, to_test) < 0.085 or cosine(sample_w3, to_test) < 0.085 or cosine(sample_w4, to_test) < 0.085:
        remove(screen)
        return 1
    elif cosine(sample_l, to_test) < 0.075:
        remove(screen)
        return 2
    else:
        remove(screen)
        return 0

def localize_me():
    screen = take_screenshot()
    if len(locate_sub_image("templates/btn_map.png", screen)) or len(locate_sub_image("templates/btn_map_2.png", screen)):
        ret = "home"
    elif len(locate_sub_image("templates/btn_home.png", screen)) or len(locate_sub_image("templates/btn_home_2.png", screen)):
        ret = "map"
    elif len(locate_sub_image("templates/is_in_battle.png", screen)) or len(locate_sub_image("templates/is_in_battle_2.png", screen)):
        ret = "battle"
    remove(screen)
    return ret

def avoid_spam():
    screen = take_screenshot()
    if tap_on_template_found('templates/spam_open_now.png', screen):
        time.sleep(10)
        if not tap_on_template_found('templates/spam_ok.png', 0):
            launch_cmd("adb shell input tap {} {}".format(400 / RES_SCALE, 355 / RES_SCALE))
            time.sleep(10)
            tap_on_template_found('templates/spam_ok.png', 0)
            time.sleep(2)
            if not tap_on_template_found('templates/btn_close.png', 0):
                tap_on_template_found('templates/spam_ok.png', 0)
            time.sleep(2)
    elif tap_on_template_found('templates/spam_ok.png', 0):
        time.sleep(2)
    elif tap_on_template_found('templates/btn_close.png', 0):
        time.sleep(2)
    elif tap_on_template_found('templates/spam_claim.png', 0):
        time.sleep(10)
        tap_on_template_found('templates/spam_ok.png', 0)
        time.sleep(3)
    elif cosine_sim('templates/been_attacked.png'):
        launch_cmd("adb shell input tap %d %d" % (400 / RES_SCALE, 400 / RES_SCALE))
        time.sleep(2)
    remove (screen)

def save_battle(camp_name, to_append):
    with open('battle_info.json', 'r') as f:
        data = json.load(f)
    data[camp_name] = to_append

    with open('battle_info.json', 'w') as f:
        json.dump(data, f)

def update_percentage(file_names, percs):
    with open('battle_info.json', 'r') as infile:
        data = json.load(infile)

    i = 0
    for r in file_names:
        data[r]['destr_perc'] = percs[i]
        i += 1
    with open('battle_info.json', 'w') as outfile:
        json.dump(data, outfile)

def pinch_out():
    with open('sendevent_input_nexus5.sh') as f:
        for l in f:
            if l[0] != '#':
                launch_cmd("adb shell " + l)


########################### MAIN ############################
def main():
    # try:
    global WHERE_I_AM
    global data
    while 1:
        go_on = True
        avoid_spam()
        WHERE_I_AM = localize_me()
        print (WHERE_I_AM)

        screen = take_screenshot()
        if WHERE_I_AM == "home":
            # tap on map button
            if not tap_on_template_found('templates/btn_map.png', screen):
                tap_on_template_found('templates/btn_map_2.png', screen)
            time.sleep(6)   # waiting for the end of the animation
            #   and tap on battle button
            tap_on_template_found('templates/btn_battle_map.png', 0)
            print("Loading village...")
            load_village()

        elif WHERE_I_AM == "map":
            tap_on_template_found('templates/btn_battle_map.png', screen)
            print("Loading village...")
            load_village()
        elif WHERE_I_AM != "battle":
            print("where I am?")
        remove(screen)

        battle_cont = 0
        completed_battles = []
        while go_on:
            ended = False
            data = {}

            ellipse, cont, stronghold, camp_file_name = locate_village()
            (data['ellipse_center_x'], data['ellipse_center_y']), (data['ellipse_len_x_axis'], data['ellipse_len_y_axis']), data['ellipse_angle'] = ellipse
            (data['stronghold_x'], data['stronghold_y'], data['stronghold_level']) = stronghold
            launch_troops(ellipse, cont, data['stronghold_x'], data['stronghold_y'], camp_file_name, data)
            print("\nWaiting for the end of the battle...")
            while not ended:
                res = end_battle()
                if res == 1:
                    ended = True
                    data['battle_result'] = 1
                    data['destr_perc'] = 100
                    print("VICTORY!!!")
                elif res == 2:
                    ended = True
                    data['battle_result'] = 0
                    data['destr_perc'] = 20
                    print("DEFEAT!")
                else:
                    time.sleep(3)
            save_battle(camp_file_name, data)
            battle_cont += 1
            completed_battles.append(camp_file_name)
            print (data)
            time.sleep(3)
            if tap_on_template_found('templates/btn_battle_another.png', 0):
                print("Loading village...")
                load_village()
            else:
                go_on = False


        tap_on_template_found('templates/btn_continue.png', 0)
        time.sleep(10)

        while not cosine_sim('templates/map_after_battles.png'):
            avoid_spam()
        time.sleep(5)
        if not tap_on_template_found('templates/btn_home.png', 0):
            tap_on_template_found('templates/btn_home_2.png', 0)
        while not len(locate_sub_image("templates/btn_map.png", 0)) + len(locate_sub_image("templates/btn_map_2.png", 0)):
            avoid_spam()
        time.sleep(5)
        avoid_spam()
        tap_on_template_found('templates/btn_menu.png', 0)
        time.sleep(10)
        tap_on_template_found('templates/btn_battle_history.png', 0)
        time.sleep(5)
        percs = take_percs(battle_cont)
        update_percentage(completed_battles, percs)
        tap_on_template_found('templates/btn_close.png', 0)
        time.sleep(5)
        tap_on_template_found('templates/btn_close.png', 0)
        time.sleep(5)
        if not tap_on_template_found('templates/btn_map.png', 0):
            tap_on_template_found('templates/btn_map_2.png', 0)
        for i in range (0,60):
            print ("{} mins to go!".format(60 - i))
            time.sleep(60) # 20 mins: 60 secs * 20

        print ("done.")

    # except Exception as e:
    #     with open('battle_info.json', 'r') as f:
    #         data = json.load(f)
    #     a, b = progressive_nameFile('camp_screen/')
    #     current_camp = 'camp_screen/camp{}.png'.format(b-1)
    #     print (e)
    #     try:
    #         a = data[current_camp]
    #     except Exception as e:
    #         print ("ERROR! \n Screenshot saved but no data in battle_info.json. Removing the screen...")
    #         remove(current_camp)
    #         main()


######################### FOR_DEBUG #########################
def debug():

    ############### DEBUG PER PERCENTUALI ##################
    # y = STATS_Y
    # ret = [21, 41, 8, 38, 48, 26, 38, 31, 8, 19, 3]
    # cont = 0
    # print (ret)
    # for i in range(11):
    #     res = perc_recognition('{}.png'.format(i+1))
    #     print("Real: {}     res: {}".format(ret[i], res))
    #     if res == ret[i]:
    #         cont +=1
    # print ('Correct: {}     Errors: {}'.format(cont, 11 - cont))
    ########################################################

    #################### DEBUG VARIO ########################
    # print(take_screenshot())
    # while not cosine_sim('templates/map_after_battles.png'):
    #     avoid_spam()
    #
    # locate_sub_image('templates/btn_map.png', 0)
    # avoid_spam()
    # print(take_percs(4))
    # print(perc_recognition(0))
    ########################################################

    ################# DEBUG PER VIDEO ######################

    data = {}
    ellipse, cont, stronghold, camp_file_name = locate_village()
    (data['ellipse_center_x'], data['ellipse_center_y']), (data['ellipse_len_x_axis'], data['ellipse_len_y_axis']), data['ellipse_angle'] = ellipse
    (data['stronghold_x'], data['stronghold_y'], data['stronghold_level']) = stronghold
    launch_troops(ellipse, cont, data['stronghold_x'], data['stronghold_y'], camp_file_name, data)


    ########################################################


if __name__ == "__main__":
    main()
    # debug()
