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

* So the first step was to get rid of these duplicates... but some were "only" near-duplicates (not exact duplicates). I tried several tools and was not very happy with the results

* Once again I decided to write my own tool : image-match. I am pretty proud of this one, because:
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
* You can repeat this task many times, images are added only once per folder
* You can "Clear" the images list to start a new list
* You can check/uncheck all images, or some of them using the keyword search, or check/uncheck only the selected images, and even invert the selection
* You can hide the checked images, or even delete the corresponding files
* if you double-click an image with the mouse, a new window will appear with a bigger version + some information

### "DUPLICATES" TAB

![Screenshot - Image](screenshots/screenshot-image.jpg?raw=true)

This is the heart of the program. Here you can look for image similarities, using several algorithms.

* 1. select the algorithm you want to use
* 2. select the level of similarity or directly enter a percentage
* 3. click on "Compare images"
* 4. some algorithms are super fast (all the hash-based ones), others will be much slower. Just wait for the operation to finish
* 5. now the duplicates are displayed in groups
* 6. some things you can try:
   * double-clicking and image shows it in a new window, at bigger size
   * check 2 images and click on the magnifying glass button: a new window will appear with the 2 images, the exact percentage of similarity for this pair and for the current algorithm. Chances are you will also see matched features (cyan lines), and even an homothety if detected. This enables you to understand a bit more how the images are correlated
   * you can check/uncheck images, even with a keyword search. You can also select the first image in each group, or all the images but the first one in each group
   * you can also copy / move / delete the checked image files

### "OPTIONS" TAB

![Screenshot - Image](screenshots/screenshot-image.jpg?raw=true)

Not a lot to see here, default values should work fine.

Anyway if you want bigger or smaller image icons, change the "thumbnails size" when the images list is empty (option not available if images are already shown).

Second option: "Image work size". All images are reduced to this amount (in pixels) internally, before feeding them to the algorithms. 256 pixels is a good tradeoff between speed and accuracy.

The last option is for the "Features" and "Homography" algorithms: more features can be looked for, the operations will be more accurate but also slower.

The big blocks with algorithms names are for the special similarity mode "Combined". More about it later.

### IMAGES CLUSTERING

The Duplicates tab displays image matches. But how were they regrouped?

* Good to know, all algorithms give results as a percentage of similarity, from 0 to 100%.
* Each image pair has a score of x% for a given algorithm.
* First pass: all the images are tested with each other. This means the more images you have in the images tab, the more operation there will be, because the number of matches will be N.(N-1) / 2 - for example if you have 15000 images to test, there will be 105 million tests to perform!
* After the 1st pass, all image pairs have a score. Each image has a list of its similar images
* Second pass: look for each image's closest neighbour. If this neighbour also has a closest neighbour, compare the scores and decide which image goes with which one. No image can be added to a group if its score with all images in this group isn't over the threshold
* Last pass: identify the leftovers (some images were not regrouped) and decide in which group they should be attached to. Constraint are less restrictive

### IMAGE SIMILARITY ALGORITHMS

* OpenCV has a ready-to-use bunch of algorithms:
   * aHash [Average Hash]: not very interesting
   * pHash [Perceptual Hash]: 

## Enjoy!

### AbsurdePhoton
My photographer website : www.absurdephoton.fr

All the photos used on this page were shot by AbsurdePhoton, they are the only copyrighted elements of "image-match".
