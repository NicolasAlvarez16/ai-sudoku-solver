from imutils.perspective import  four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

def find_puzzle(image, debug=False):
    # Convert the image to gray scale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3) # 7 x 7 Kernel

    # Apply adaptative thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # Check if we are visualising each step of the image processing pipeline (in this case, thresholding)
    if debug:
        cv2.imshow('Puzzle Thresh', thresh)
        cv2.waitKey(5)
    
    # find the contours in the hresholded image and sort them by size in descending order
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Initialise a contours that correspondes to the puzzle outline
    puzzle_count = None

    # Loop over the contours
    for c in contours:
        # Approximate the contours
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If approximate countor has four points then we can assume that we have found the outline of the puzzle
        if len(approx) == 4:
            puzzle_count = approx
            break
    
    # If the puzzle contour is empty then our script could not find the outline of the Sudoku puzzle so raise an error
    if puzzle_count is None:
        raise Exception(('Could not find the Sudoku puzzle outline. Try debugging your threshold and contour steps.'))
    
    # Check if we are visualising the outline of the detected Sudoku puzzle
    if debug:
        # Draw the contour of the puzzle on the image and then disply it for visulaisation/debuggine purposes
        output = image.copy()
        cv2.drawContours(output, [puzzle_count], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(5)
    
    # Apply a four point perspective transform to both the original image and a grayscale image to obtain a top-down bird's eye view of the puzzle
    # Applying a four-point perspective transform effectively deskews our Sudoku puzzle grid, making it much easier for us to determine rows, columns, and cell
    puzzle = four_point_transform(image, puzzle_count.reshape(4, 2))
    warped = four_point_transform(gray, puzzle_count.reshape(4, 2))

    # Check to see if we are visualising the perspective transform
    if debug:
        # Show the output warped image (again, for debugin purposes)
        cv2.imshow('Puzzle Transform', puzzle)
        cv2.waitKey(5)
    
    # Return a 2 tuple of puzzle in both RGB and gray scale
    return (puzzle, warped)

def extract_digits(cell, debug=False):
    # Apply automatic thrsholfing to the cell and then clear any connected borders that touch the border of the cell
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    # Check to see if we are visualising the cell thresholding step
    if debug:
        cv2.imshow('Cell Thresh', thresh)
        cv2.waitKey(5)
    
    # Find the countors of the threshold cell
    countors = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    countors = imutils.grab_contours(countors)

    # If no countors were found this cell is empty
    if len(countors) == 0:
        return None
    
    # Otherwise finde the largest counter in the cell and creare a mask for the cell
    c = max(countors, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype='uint8')
    cv2.drawContours(mask, [c], -1, 255, -1)

    # Computer the percentage of masked pixels relative to the total area of the image
    (h, w) = thresh.shape
    percentage_filled = cv2.countNonZero(mask) / float(w * h)

    # if less than 3% of the mask is filled then we are looking at noise and can safely ignore the countor
    if percentage_filled < 0.03:
        return None
    
    # Apply the mask to the threshold cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    # Check if we should visualise the masking step
    if debug:
        cv2.imshow('Digit', digit)
        cv2.waitKey(6)
    
    return digit


    