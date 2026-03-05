command="./path_select.py"

for file in data/testing/*.json
do
    $command "$file"
done

for file in data/*.json
do
    $command "$file"
done
