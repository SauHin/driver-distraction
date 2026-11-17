# Guide for the own recordings and the live test

This guide tells you where to put the camera and how to record each person.
`docs/PLAN.md` sections 3.3, 9, and 10 give the rules. Decisions D21 and D22 apply.

## Safety and consent

- Record in a parked car only. The engine is off, and the parking brake is on.
- Do not record a person in a car that moves.
- Each person signs the consent form before the first recording. See `consent_form.md`.
- A person can stop at any time. Then delete the files of that person.

## The car and the camera

- The car of the project has the steering wheel on the right (right-hand drive).
- The StateFarm cars have the steering wheel on the left. Their camera sits on the passenger side and sees the driver from the side.
- Thus put the camera on the passenger side, on the left of the driver.
- The view is then a mirror of the StateFarm view. Run the app with `--mirror` (decision D21).
- The camera is a phone. For the live test, the phone is a webcam of the laptop (decision D22).

## The camera position

Copy the StateFarm view as well as you can:

1. Put the phone at the passenger seat, at about the head height of a passenger.
2. Aim the phone at the driver from the side. Turn it a little toward the front of the car.
3. Hold the phone on its side, so the frame is wider than it is high. Fix the phone with a clamp or a stand.
4. Make sure that the frame shows the head, the upper body, both hands, and the steering wheel of the driver.
5. Make sure that the steering wheel is near the side of the frame that points to the front of the car.
6. If the phone has a 4:3 video size, use it. StateFarm images are 640 by 480 pixels, a 4:3 frame.

If the video is wider than 4:3, put the driver in the middle of the frame.
The app cuts a square from the middle of each frame. Thus a wide frame loses more of its sides.

## Light

- Record in daylight. If possible, park the car in shade.
- Do not aim the camera into the sun.
- Do not use a light that points at the face of the driver.

## Check the view before you record

1. Connect the phone as a webcam. Its app gives the laptop a camera number, for example 1.
2. Start the app with the mirror option: `.venv/Scripts/python app/live.py 1 --mirror`.
3. The driver sits still with both hands on the wheel. The status must show "SAFE".
4. If the status is wrong, move the phone, and try again.
5. Press q to stop the app.

If the number 1 does not open the phone, try 2 or 0.

## The recordings of each person

Each person does two recordings. The labels use the real hand of the person.

1. Ten short clips, one clip for each class. Each clip is about 10 seconds (decision P19).
2. One continuous recording. The person starts with safe driving. Then the person does each action for about 10 seconds, with safe driving between the actions.

| Code | Action |
|---|---|
| `c0`. | Safe driving: both hands on the wheel, eyes to the front. |
| `c1`. | Texting with the right hand. |
| `c2`. | A phone call with the right hand. |
| `c3`. | Texting with the left hand. |
| `c4`. | A phone call with the left hand. |
| `c5`. | Use of the radio or the screen of the car. |
| `c6`. | Drinking. |
| `c7`. | Reaching behind. |
| `c8`. | Hair and makeup. |
| `c9`. | Talking to a passenger. |

For `c9`, a helper sits behind the passenger seat, out of the view of the camera. The driver turns the head and talks to the helper.

## The label file

Write one CSV file for each video. The file has the same name as the video, with `_labels.csv` at the end.
`labels_example.csv` shows the columns:

- `start_seconds`: the start of the action in the video.
- `end_seconds`: the end of the action in the video.
- `class`: the StateFarm code of the action, by the class name and the real hand.

The app reads the start and the class. Use `--truth` to show the true class on the screen.

## Where the files go

- Put the files of each person in `data/own/<person code>/`. Use codes such as `s01`, not names.
- The folder `data/` is not in git. Do not put the recordings on a public drive.
- Keep the signed consent forms on paper or in a private folder.

## Persons and the split

- No person is in two parts of a split.
- With 6 persons or more, half of the persons select the smoothing values. The other half is the test.
- With fewer persons, the smoothing values stay at the values of decision P18. All persons are the test.
