from py_image_search.sudoku.puzzle import extract_digits
from py_image_search.sudoku.puzzle import find_puzzle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import numpy as np
import argparse
import imutils
import cv2

# Construct the argument parser and parse the arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-m', '--model', required=True, help='path to train digital classifier')
arg_parser.add_argument('-i', '--image', required=True, help='path to input Sudoku puzzle image')
arg_parser.add_argument('-d', '--debug', type=int, default=-1, help='weather or not we are visualising each step of the pipeline')
args = vars(arg_parser.parse_args())

# Load digit classifier from disk
print('[INFO] loading digit classifier...')
model = load_model(args['model'])

# Load the input image from disk and resize it
print('[INFO] processing image..')
image = cv2.imread(args['image'])
image = imutils.resize(image, width=600)

# Finde puzzle in the iamge
(puzzle_image, warped) = find_puzzle(image, debug=args['debug'])

# Initialise grid 9x9
board = np.zeros((9, 9), dtype='int')

# We can infer the location of each cell by diving the warped image into a 9x9 grid
stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9

cell_coords = [] # Initialise a list to store the x and y coordinates of each cell

for y in range(0, 9):
    row = [] # Initialise the current list of cell locations

    for x in range(0, 9):
        # Compute the current x and y cordinates of the current cell
        startX = x * stepX
        startY = y * stepY
        endX = (x + 1) * stepX
        endY = (y + 1) * stepY

        # Add the x and y coordinates to the cell location list
        row.append((startX, startY, endX, endY))

        # Crop the cell from the warped transform image and then extract the digit from the cell
        cell = warped[startY:endY, startX:endX]
        digit = extract_digits(cell, debug=args['debug'])

        # Verify that the digit is not empty
        if digit is not None:
            # Resize the cell to 28x28 pixels and then prepare the cell for classification
            roi = cv2.resize(digit, (28, 28))
            roi = roi.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Classify the digit and update the Sudoku board with the prediction
            prediction = model.predict(roi).argmax(axis=1)[0]
            board[y, x] = prediction
    cell_coords.append(row)

# Construct a Sudoku puzzle from the board
print("[INFO] OCR'd Sudoku board:")
puzzle = Sudoku(3, 3, board=board.tolist())
puzzle.show()

# Solve the Sudoku puzzle
print("[INFO] Solving the Sudoku Puzzle...")
solution = puzzle.solve()
solution.show_full()

# Loop over the cell locations and board
for (cell_row, board_row) in zip(cell_coords, solution.board):
    # Loop over individual cell in the row
    for (box, digit) in zip(cell_row, board_row):
        # Unpack the cell coordinates
        start_x, start_y, end_x, end_y = box

        # Computr the coordinates of where the digit will be drawn on the output image
        text_x = int((end_x - start_x) * 0.33)
        text_y = int((end_y - start_y) * -0.2)
        text_x += start_x
        text_y += end_y

        # Draw the result digit on the sudoku puzzle iamge
        cv2.putText(puzzle_image, str(digit), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

cv2.imshow('Sudoku Result', puzzle_image)
cv2.waitKey(100000000)

