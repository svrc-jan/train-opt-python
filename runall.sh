command="./preprocess.py"

for file in data/*.json
do
    $command "$file"
done
