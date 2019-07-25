for file in */*.txt
do
data=$(cat $file | grep SER)
echo "$data" > $file.res
done 