select concat(likes,',',shares,',',comments,',',hashtags,',',images,',',vdo,',',url,',',word_in_dict,',', word_outside_dict,',',num_of_number_in_sentense,',',app_sender,',',share_with_location,',',share_with_non_location,',',tag_with,',',feeling_status,',',share_public,',',share_only_friend,',',word_count,',',character_length,',',question_mark,',',exclamation_mark) from training_data into outfile '/tmp/training.txt';

select cred_value
from training_data
into outfile '/tmp/result.txt'