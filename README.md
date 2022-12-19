# image-match-opencv
## A nice GUI using Qt and OpenCV to perform images similarity matching from entire folders - several CUSTOM ALGORITHMS - optimized to be incredibly EFFICIENT and FAST
### v0 - 2022-12-0618

![Screenshot - Global](screenshots/match.jpg?raw=true)
<br/>
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

![Screenshot - Images tab](screenshots/images-tab.jpg?raw=true)
<br/>

* Just click the "Add images" button: choose the folder containing the images to compare - many image formats are supported (15+)
* Folders can be crawled recursively by checking the checkbox near this button
* You can repeat this task many times, images are added only once per folder
* You can "Clear" the images list to start a new list
* You can check/uncheck all images, or some of them using the keyword search, or check/uncheck only the selected images, and even invert the selection
* You can hide away (exlude) images, or even delete the corresponding files
* if you double-click an image with the mouse, a new window will appear with a bigger version + some information

### "DUPLICATES" TAB

![Screenshot - Duplicates tab](screenshots/duplicates-tab.jpg?raw=true)
<br/>

This is the heart of the program. Here you can look for image similarities, using several algorithms.

* 1. select the algorithm you want to use
* 2. select the level of similarity or directly enter a percentage
* 3. click on "Compare images"
* 4. some algorithms are super fast (all the hash-based ones), others will be much slower. Just wait for the operation to finish
* 5. now the duplicates are displayed in groups
* 6. some things you can try:
   * double-clicking an image shows it in a new window, at bigger size
   * check 2 images and click on the magnifying glass button: a new window will appear with the 2 images, the exact percentage of similarity for this pair and for the current algorithm. Chances are you will also see matched features (cyan lines), and even an homothety if detected. This enables you to understand a bit more how the images are correlated
   * you can check/uncheck images, even with a keyword search. You can also select the first image in each group, or all the images but the first one in each group
   * you can now copy / move / delete the checked image files

### "OPTIONS" TAB

![Screenshot - Options tab](screenshots/options-tab.jpg?raw=true)
<br/>

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
* Last pass: identify the leftovers (some images were not regrouped) and decide to which group they should be attached. Constraint are less restrictive

### IMAGE SIMILARITY ALGORITHMS

![Screenshot - Algorithms](screenshots/algorithms.jpg?raw=true)
<br/>

* OpenCV has a ready-to-use bunch of algorithms:
   * aHash (Average Hash): not very interesting - not used here
   * pHash (Perceptual Hash): basic but effective and fast!
   * Block Mean Hash: grayscale means are used
   * Marr-Hildreth Hash: this operator is used then binarized
   * Radial Variance Hash: this one can detect rotations to some extent
   * Color moments: this one gives too much false positives, this is why I thought about using dominant colors, more about this in a moment...
* I added myself these ones:
   * dHash (Difference Hash): a tiny 9x8 pixels version of the image is used and pixels are compared with luminosity changes
   * idHash (Important Difference Hash): same principle as dHash, but horizontal AND vertical scans are performed on a 9x9 pixels tiny version of the original image
   * Dominant Colors: the dominant color of each image (it is NOT a mean) is computed, then these values are compared using their distance in the OKLAB color space - this way images are regrouped by "global" color - not very accurate but very useful for the special similarity mode "Combined" - notice that the dominant color algorithm is of my own design, called "Sectored-Means"
   * DNN Classify: some AI is used here, and you better have a NVidia GPU, although computing with CPU is supported (much slower). Images are classified using a 21K classes reference, and then are compared using the most used percentages of the matched classes - not very accurate but useful for the special similarity mode "Combined" - you'll have to download a big 128MB Caffe model file (with a BitTorrent client) to be able to use it
   * Features: images features are matched between the pairs, the more they have in common the more the score will be. This method is able to detect extremely rotated versions of an image - this is very SLOW and you should use it on reduced images lists (2K-3K max)
   * Homothety: a step further from "Features", if a sufficent number of "good" matches are found, an homothety could be found - this usually means images are similar. This method can detect not-so-near duplicates, and extremely rotated versions - this is very efficient but also very SLOW, and you should use it on reduced images lists (2K-3K max)
* this tool is not perfect:
   * mirrored images are not easy to find, and probably won't be listed in the duplicates groups, unless you're using methods like "Dominant Color" and "DNN Classify" + features in a COMBINED way
   * extreme threshold values will surely produce many false-positives 
* Also notice:
   * a lot of results are cached when an algorithm is used: you can recompute the same algorithm with a different threshold in a very reduced time compared to the first pass!
   * with 48GB of RAM, you can test about 25K images, but it is not a good idea to do that in a unique pass (long wait). Prefer sub-groups!

### SPECIAL "COMBINED" SIMILARITY ALGORITHM

* When you use an algorithm, it will be "activated" in the "Options" tab. If you run several algorithms on the same images set, their respective option will be activated
* You can now COMBINE several algorithms to find even more matches between your images!
* To do that, in the "Options" tab check/uncheck some algoritms
* In the "Duplicates tab, select "Combined" as the algorithm to use
* Select a high similarity threshold, like 50% - run it - if all goes well more matches should be found, some of them could even surprise you - it is not perfect of course but sometime it is very efficient - just play with different threshold values, results are already cached so computation will be very fast in the "Combined" mode
* What algorithms to use together? It's easy: use "idHash" alongside others that are not very accurate by themselves like "DNN classify", "Dominant Colors", plus some using like "Features" and "Homography" - for example I often use a threshold of 35% with "DNN" + "Dominant colors" + "Homography" + "idHash"
* How is a "combined" score computed? Each result for each pair of images for each algorithm is classified as "Exact", "Similar", Different" or "Dissimilar" - this gives a weight that can be multiplied with the initial score - the averaged sum of all algorithms scores is then calculated to give the final result. This looks almost too easy, but it definitely works!

### MAYBE A TO-DO LIST

* If sufficient attention is given to this program, maybe I could take the time to make it even harder, better, faster, stronger (yes this a song!)
* make the slowest similarity algorithms run on GPU (all ?)
* use a database instead of keeping all operations results and cache in RAM
* with the two items above, maybe millions of images could be tested
* maybe contact "libre" image tools like Geeqie or xnView to offer them some help with similarity matching (this Geeqie feature inspired me a lot at the beginning) 

<br/>
<br/>

## Enjoy!

### AbsurdePhoton
My photographer website : www.absurdephoton.fr
