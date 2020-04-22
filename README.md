# Text-detection-in-live-video-streaming
<pre>

<h2>Python Libraries required</h2>

  imutils
  numpy
  argparse
  pytesseract
  imutils
  time
  cv2

  <h3>Our script requires command line arguments:</h3>
    --east
     : The EAST scene text detector model file path.
    --video
     : The path to our input video. Optional — if a video path is provided then the webcam will not be used.
    --min-confidence
     : Probability threshold to determine text. Optional with default=0.5
     .
    --width
     : Resized image width (must be multiple of 32). Optional with default=320
     .
    --height
     : Resized image height (must be multiple of 32). Optional with default=320
     --padding
     : The (optional) amount of padding to add to each ROI border. You might try values of 0.05
      for 5% or 0.10
      for 10% (and so on) if you find that your OCR result is incorrect.
 .

Important: The EAST text requires that your input image dimensions be multiples of 32, so if you choose to 
adjust your --width and --height values, ensure they are multiples of 32!
  
<h3>Command to run the code is</h3>
python webcam2text.py --east frozen_east_text_detection.pb 
  
</pre>
