ds=$1  # fews or wn
part=$2  # some of the options: for fews [dev, test], for wn [ALL, semeval2007]
for x in `find -name "*.ckpt"`; do 
	dir=$(dirname $x)
	encoder=$(basename $dir)
	if [[ "$ds" == "fews" ]]; then
		x="--fews-split=$part"
	elif [[ "$ds" == "wn" ]]; then
		x="--wn-split=$part"
	else
		echo "No such dataset: $ds, choose from fews and wn"
	fi

	python biencoder.py --dataset=$ds --ckpt $dir --eval "$x" --encoder-name $encoder "${@:3}" &>$dir/eval_${ds}-${part}.log

done
