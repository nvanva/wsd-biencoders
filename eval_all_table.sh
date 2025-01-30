grep -E 'f1|acc' `find -name "eval*.log"`|sed -r 's!.*weights/(.*)/eval_(.*).log.*=(.*)!\1\t\2\t\3!' | python eval_create_table.py
