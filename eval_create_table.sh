grep -E 'f1|Few-shot|Zero-shot' `find -name "eval*.log"`|sed -r 's!.*weights/(.*)/eval_(.*).log:([^ ]+).*=(.*)!\1\t\2:\3\t\4!' | python eval_create_table.py
