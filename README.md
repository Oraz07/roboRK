# RoboRK

RoboRK is a project based on Real Kingdoms Android game. The mission is to train a CNN to play with the game, in particular to fight against other users.

## Getting Started

Install Python packages with pip:
* numpy
* pytesseract

```
pip install [package_name]
```

In order to control the Android phone, ADB shell is required.
 [Adb](https://developer.android.com/studio/command-line/adb.html) is included in the Android SDK Platform-Tools package, available [here](https://developer.android.com/studio/releases/platform-tools.html).

This project works for every Android device, also emulators. To setup your device follow the next section.

### Device Setup
This procedure works for every device, also emulated.
If you want to use an Android emulator I suggest you to consider the [Android Studio emulator](https://developer.android.com/studio/run/emulator.html).
This emulator works fine for the purpose because it's possible to launch the program without UI, using `-no-window` and `-gpu off` options during startup.

To initialize a new emulator in Android Studio you have to open the `AVD Manager` and than `Create Virtual Device...`.
When you have to choose the phone I suggest you to select one of these phones, using 25 API level:
* Nexus 5
* Nexus 5X

The reason is that I already generate the pinch-out script and you can skip this step later.

I also recommend to use a Play Store pre-installed version to speedup the setup.

You can launch the emulator without UI with this command:
```
~/android-sdk-linux/tools/emulator @[name_of_the_device]  -no-window -gpu off
```
or simply delete the options to launch with UI:
```
~/android-sdk-linux/tools/emulator @[name_of_the_device]
```
or directly from the AVD Manager in Android Studio (here you can also find and modify the name of your device).

Now you have to install two apps (if you have the Play Store you can simply use it):
* [Google Play Games](https://play.google.com/store/apps/details?id=com.google.android.play.games)
* [Rival Kingdoms](https://play.google.com/store/apps/details?id=com.spaceapegames.rivalkingdoms)

To play this game is necessary to sign in with Google Play Games.
It's also essential that the game language is set on English.

### Bot Setup
Now only few steps more: we need to push to the phone a script necessary to generate the pinch out (or zoom out) gesture.
If you use one of the recommended smartphones you already have the generated script file in the repository.
Simply use one of this files:
* sendevent_input_nexus5.sh
* sendevent_input_nexus5X.sh

copying it with the general file name with this command:
```
cp sendevent_input_[device].sh sendevent_input.sh
```
Otherwise if you're using another device you have to follow the steps described [here](https://stackoverflow.com/questions/25363526/fire-a-pinch-in-out-command-to-android-phone-using-adb#25629952) to generate the file.
To push the file inside the phone it's possible to use this ADB command:
```
adb push sendevent_input.sh /sdcard/
```
It's important to maintain this file name for the script due to grant the program functionality.

Last but not least you need to inform the bot about your troops that you use in battle.
You can do this with two files:
* `troops.json`: in this file you have to specify name, number of elements and level of every troop that you decide to use.
Note that also the dragon (available only from stronghold level 4) is a troop instead of the ancient that is not considered in this project.

```
{
  "troops": [
    {
      "angle": 0,
      "count": 3,
      "level": 1,
      "name": "soldier",
      "x": 0,
      "y": 0
    },
    ...
}
```
* `background.png`: this is the sample image used to extract the contour of the village to attack.
You need to use the implemented function `take_screenshot()` in the robork.py file to return the right resolution of the image while you're about to attack another village. Note that is really important to zoom out at max and locate te view on the right bottom corner of the screen (swipe from right lo left and from bottom to the top). You can easly launch the bot and take the first camp screen obtained.

Now your device is ready to play!

## Built With

* [Python](https://www.python.org/) - Main programming language
* [openCV](https://opencv.org/) - Open Source Computer Vision Library
* [NumPy](http://www.numpy.org/) - Fundamental package for scientific computing with Python
* [pytesseract](https://pypi.python.org/pypi/pytesseract/0.1) - Optical character recognition (OCR) tool for python
* [Keras](https://keras.io/) - High-level neural networks API
* [Android Debug Bridge] (https://developer.android.com/studio/command-line/adb.html) - Command-line tool that lets you communicate with a device
* [Android Studio emulator](https://developer.android.com/studio/run/emulator.html) - Android emulator

## Authors

* **Gabriele Orazi** - *Main developer* - [oraz07](https://github.com/oraz07)
* **Andrea Asperti** - *Professor* - [asperti](https://github.com/asperti)
