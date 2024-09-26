system='''Task: Each video consists of n events, and the text description of each event has been given correspondingly (separated by " ",). You need to judge whether the first n-1 events in the video are the cause of the last event, the probability of the cause 0(irrelevant) or 1(relevant) is expressed as the output.
            Here are several example of judging whether the first n-1 events in the video are the cause of the last event:
            First example: Text description of 6 events:\
            A man wearing a black clothes is sharpening the knife on a stone,\
            The man beside him on blue long sleeves drawing something on the table with his finger,\
            The man turn the knife to sharpen the other side and then wipe it with paper towel,\
            The man in navy blue shirt point to the knife while the man sharpening the knife tries to sharpen it one hand,\
            He continues sharpening the knife, turn it again to further sharpen the other side and wipe it again with paper towel,\
            Throws the old and dirty paper towel and reach the roll of paper towel and clean the knife.\
            The probability output should be (length should be 5): [0, 0, 1, 0, 1]
            Second example: Text description of 9 events: \
            A woman holds a yellow ball behind her neck,\
            She turns around and launches the ball onto a field,\
            People run to measure the distance,\
            A man in a red shirt holds a ball behind his neck,\
            He turns around and launches the ball on the field,\
            People run to measure the distance of it,\
            Another woman holds a ball behind her neck,\
            She also turns around and launches the ball onto the field,\
            People then run over to measure the distance.\
            The probability output should be (length should be 8): [0, 0, 0, 0, 0, 0, 1, 1]
            Third example: Text description of 9 events:\
            The white water polo team huddles together,\
            The game begins and blue scores while deep in the white defense to tie the game,\
            White answers with a powerful goal to grab the lead back,\
            Blue moves in and after a couple of passes ties the game back up,\
            White scores to take the lead, and blue comes right back to tie once again,\
            A scramble for the ball results in white gaining a 2-on-1 advantage and a goal,\
            They push their lead further late and put the game out of reach, winning 14-10,\
            The team celebrates together in the water,\
            The team celebrates their gold metal at the podium.\
            The probability output should be (length should be 8): [0, 0, 1, 0, 1, 1, 1, 0].
            The assistant should give probability output to the user's input video descriptions.'''
print(system)

system="Task: Each video consists of n events, and the text description of each event has been given correspondingly (separated by " ",). You need to judge whether the first n-1 events in the video are the cause of the last event, the probability of the cause 0(irrelevant) or 1(relevant) is expressed as the output.\
            Here are several example of judging whether the first n-1 events in the video are the cause of the last event:"

print(system)