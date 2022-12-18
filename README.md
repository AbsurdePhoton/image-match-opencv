# image-match-opencv
## A nice GUI using Qt and OpenCV to perform images similarity matching from entire folders - several algorithms - optimized to be incredibly EFFICIENT and FAST
### v0 - 2022-12-0618

![Screenshot - Global](screenshots/screenshot-gui.jpg?raw=true)
<br/>

## HISTORY

* v0: launch
<br/>
<br/>

## LICENSE

* The present code is under GPL v3 license, that means you can do almost whatever you want with it!

* I SOMETIMES used bits of code from several sources, which are cited within the code
<br/>
<br/>

## WHY?

* Lately I wanted to organize many image files I had downloaded from the internet since many years. They are all from other photographers, most are really inspiring and can give me ideas. But there were 25K of them, and there were many duplicates!

* So the first step was to get rid of these duplicates... but some were "only" near-duplicates (not exact duplicates). I tried several tools and was not very happy with the results.

* Once again I decided to write my own tool : image-match

* I am pretty proud of this one, because:
    * It required a lot of knowledge acquired these last years in many fields, since I published my first tool on gitHub
    * I was able to write a very EFFICIENT and FAST tool, designing a lot of opitmizations (about half the time spent on it)
<br/>
<br/>

## WITH WHAT?

Developed using:
* Linux Ubuntu	22.04
* QT Creator 6.0.2
* Requires these libraries:
    * QT 6
    * openCV 4.6 compiled with openCV-contribs - may work with any 4.x version
* A c++ compiler which supports c++17

This software should also work under Microsoft Windows, with adjustments: if you compiled it successfully please contact me, I'd like to offer compiled Windows executables too
<br/>
<br/>

## HOW?

* Help is available within the GUI, with each element's tooltip. Just use the "What's this?" button

* All is on one window. The tool is not very complex with its 3 tabs

### "IMAGES" TAB

![Screenshot - Image](screenshots/screenshot-image.jpg?raw=true)

* Just click the "Add images" button: choose the folder containing the images to compare

* Folders can be crawled recursively by checking the checkbox near this button

* You can repeat this task many times, duplicates are eliminated

### THE REST OF THE TUTORIAL IS A WORK IN PROGRESS...


## Enjoy!

### AbsurdePhoton
My photographer website : www.absurdephoton.fr

All the photos used on this page were shot by AbsurdePhoton, they are the only copyrighted elements of "image-match".
