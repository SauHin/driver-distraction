# User test of the application

This file holds the plan and the forms of the user test (`docs/PLAN.md` section 10).
The guideline asks for the prediction interpretation, the workflow, the practical limitations, and the user satisfaction.
Use the guide `GUIDE.md` for the camera position and the safety rules.

## Plan of one session

1. The person reads and signs the consent form (`consent_form.md`).
2. The person sits in the seat of the driver of the parked car. The engine is off.
3. The tester starts the app with the phone as the webcam: `.venv/Scripts/python app/live.py 1 --mirror`.
4. The person does the actions of the action form, in its order. Each action is about 10 seconds (decision P19).
5. The tester marks the result of each action on the action form.
6. The person fills in the question form.
7. The tester keeps the log file of the app from `runs/live_logs/`. Its name goes on the action form.

All answers are real. Do not fill in a value for a person who did not do the test.

## Action form

| Field | Value |
|---|---|
| Person code. |  |
| Date and time. |  |
| Car and light. |  |
| Log file of the app. |  |

For each action, mark the status that the app shows, and whether the alert started.

| Step | Action | Code | Status on the screen | Alert started | Notes |
|---|---|---|---|---|---|
| 1. | Safe driving. | `c0`. |  |  |  |
| 2. | Texting with the right hand. | `c1`. |  |  |  |
| 3. | Safe driving. | `c0`. |  |  |  |
| 4. | A phone call with the right hand. | `c2`. |  |  |  |
| 5. | Safe driving. | `c0`. |  |  |  |
| 6. | Texting with the left hand. | `c3`. |  |  |  |
| 7. | Safe driving. | `c0`. |  |  |  |
| 8. | A phone call with the left hand. | `c4`. |  |  |  |
| 9. | Safe driving. | `c0`. |  |  |  |
| 10. | Use of the radio or the car screen. | `c5`. |  |  |  |
| 11. | Safe driving. | `c0`. |  |  |  |
| 12. | Drinking. | `c6`. |  |  |  |
| 13. | Safe driving. | `c0`. |  |  |  |
| 14. | Reaching behind. | `c7`. |  |  |  |
| 15. | Safe driving. | `c0`. |  |  |  |
| 16. | Hair and makeup. | `c8`. |  |  |  |
| 17. | Safe driving. | `c0`. |  |  |  |
| 18. | Talking to a passenger. | `c9`. |  |  |  |
| 19. | Safe driving. | `c0`. |  |  |  |

## Question form

Give a mark from 1 to 5 for each statement. 1 is "I do not agree at all". 5 is "I agree fully".

| Statement | Mark |
|---|---|
| The app was easy to start. |  |
| I understood the status on the screen. |  |
| I understood the class on the screen. |  |
| The alert came at the right time. |  |
| During safe driving, the app gave no alert. |  |
| The camera position did not disturb me. |  |
| I find such an alert useful in a real car. |  |

Short answers:

| Question | Answer |
|---|---|
| What did the app do well? |  |
| What did the app do wrong? |  |
| Which change do you want first? |  |

## Results

- Put the filled forms of each person in `data/own/<person code>/`, with the recordings. They stay out of git.
- Report only the numbers of real sessions. The report gives the number of persons with each result.
