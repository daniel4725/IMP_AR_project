import cv2

class state_machine():
    def __init__(self):
        self.state = 'start'

    def transition(self, nextstate):
        self.state = nextstate


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(0)
gui_state = state_machine()
roi = [140, 360, 400, 620]  # [y_start, y_end, x_start, x_end]
# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    font = cv2.FONT_HERSHEY_PLAIN

    # Use putText() method for
    # inserting text on video
    if gui_state.state =='start':
        # ImageDraw.Draw(frame).text((120, 50),  'Hello world!',(0, 255, 255))
        cv2.putText(frame, 'For choosing state press space', (20, 20), font, 1, (125, 125 , 255), 2, cv2.LINE_4)
    if gui_state.state =='choose_app':
        cv2.putText(frame, '1. Power point', (20, 20), font, 1, (125, 125, 0), 2, cv2.LINE_4)
        cv2.putText(frame, '2. Tablet', (20, 40), font, 1, (125, 125, 0), 2, cv2.LINE_4)
    if gui_state.state =='pp':
        cv2.putText(frame, '1. insert shape', (20, 20), font, 1, (125, 125, 0), 2, cv2.LINE_4)
        cv2.putText(frame, '2. delete shape', (20, 40), font, 1, (125, 125, 0), 2, cv2.LINE_4)
        cv2.putText(frame, '3. move shape', (20, 60), font, 1, (125, 125, 0), 2, cv2.LINE_4)
    if gui_state.state =='tablet':
        cv2.putText(frame, 'Tablet mode', (20, 20), font, 1, (125, 125, 0), 2, cv2.LINE_4)
    if gui_state.state == 'insert_shape':
        cv2.putText(frame, 'Draw shape with hands', (20, 20), font, 1, (125, 125, 0), 2, cv2.LINE_4)
        cv2.rectangle(frame, (roi[2], roi[0]), (roi[3], roi[1]), (0, 255, 0), 0)  # (top left corner),(bottom right corner)
    if gui_state.state == 'delete_shape':
        cv2.putText(frame, 'touch shape to delete', (20, 20), font, 1, (125, 125, 0), 2, cv2.LINE_4)
        cv2.rectangle(frame, (roi[2], roi[0]), (roi[3], roi[1]), (0, 255, 0), 0)  # (top left corner),(bottom right corner)
    if gui_state.state == 'move_shape':
        cv2.putText(frame, 'touch shape to move', (20, 20), font, 1, (125, 125, 0), 2, cv2.LINE_4)
        cv2.rectangle(frame, (roi[2], roi[0]), (roi[3], roi[1]), (0, 255, 0), 0)  # (top left corner),(bottom right corner)

        #TODO: add the trantions
    if cv2.waitKey(1) == ord(' ') and gui_state.state == 'start':
        gui_state.transition('choose_app')
    if cv2.waitKey(1) == ord('2') and gui_state.state == 'choose_app':
        gui_state.transition('tablet')
    if cv2.waitKey(1) == ord('1') and gui_state.state == 'choose_app':
        gui_state.transition('pp')
    if cv2.waitKey(1) == ord('1') and gui_state.state == 'pp':
        gui_state.transition('insert_shape')
    if cv2.waitKey(1) == ord('2') and gui_state.state == 'pp':
        gui_state.transition('delete_shape')
    if cv2.waitKey(1) == ord('3') and gui_state.state == 'pp':
        gui_state.transition('move_shape')
    if cv2.waitKey(1) == ord(' ') and gui_state.state == 'insert_shape':
        gui_state.transition('pp')
    if cv2.waitKey(1) == ord(' ') and gui_state.state == 'delete_shape':
        gui_state.transition('pp')
    if cv2.waitKey(1) == ord(' ') and gui_state.state == 'move_shape':
        gui_state.transition('pp')

    #Display the resulting frame
    cv2.imshow('Frame',frame)


    # Press Q on keyboard to  exit
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break
  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()

