# for file in *.tex; do
#     sed -i 's/[ \t]*$//' "$file"
# done

for file in *.tex; do
     sed -i'.bak' 's/[ ]*$//' "$file"
done
