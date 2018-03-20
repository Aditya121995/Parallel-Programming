function progress()
{
    echo "Loading all the input files into memory..."
    total=10;
    lines=`wc -l output.txt | cut -f1 -d' '`;
    sleep 5;
    echo "The task is in progress, please wait....."

    while [ $lines -le 999 ]
    do
        lines=`wc -l output.txt | cut -f1 -d' '`;

        x=$(echo "$lines/$total" | bc -l|xargs printf "%.2f")
        #x=$(($lines/$total))
        echo -ne "($x %)\r"
        sleep 0.1;

        if (($lines==1000)); then x=$(($lines*$total)) echo 'Done!' ; fi;

    done ##

}

export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64/

touch output.txt
echo "Calculating number of GPU cards in your system..."
echo
for i in {9..0};
do
    t=`CUDA_VISIBLE_DEVICES=$i ./abc.out -r 83 -c 8 -g 3`;
    x=$?;

    if(($x== 0));
    then
        break;
    fi

done

c=$(($i + 1))

echo "Number of GPUs cards in your system -->  $c"
echo


m=$1;
n=$2;
P=$3;




if(($c==1));
then
    CUDA_VISIBLE_DEVICES=0 ./LRS -r $m -c $n -s 0 -p $P -g 0 &
    progress 100 "Done"
elif(($c==2));
then
    CUDA_VISIBLE_DEVICES=0 ./LRS -r $m -c $n -s 0 -p $(($P/$c)) -g 0 &
    CUDA_VISIBLE_DEVICES=1 ./LRS -r $m -c $n -s 500 -p $(($P/$c)) -g 1 &
    progress 100 "Done"
elif(($c==3));
then
    CUDA_VISIBLE_DEVICES=0 ./LRS -r $m -c $n -s 0 -p $(($P/$c)) -g 0 &
    CUDA_VISIBLE_DEVICES=1 ./LRS -r $m -c $n -s 333 -p $(($P/$c)) -g 1 &
    CUDA_VISIBLE_DEVICES=2 ./LRS -r $m -c $n -s 666 -p $(($P/$c + 1)) -g 2 &
    progress 100 "Done"
elif(($c==4));
then
    CUDA_VISIBLE_DEVICES=0 ./LRS -r $m -c $n -s 0 -p $(($P/$c)) -g 0 &
    CUDA_VISIBLE_DEVICES=1 ./LRS -r $m -c $n -s 225 -p $(($P/$c)) -g 1 &
    CUDA_VISIBLE_DEVICES=2 ./LRS -r $m -c $n -s 500 -p $(($P/$c)) -g 2 &
    CUDA_VISIBLE_DEVICES=3 ./LRS -r $m -c $n -s 750 -p $(($P/$c)) -g 3 &
    progress 100 "Done"
elif(($c==5));
then
    CUDA_VISIBLE_DEVICES=0 ./LRS -r $m -c $n -s 0 -p $(($P/$c)) -g 0 &
    CUDA_VISIBLE_DEVICES=1 ./LRS -r $m -c $n -s 200 -p $(($P/$c)) -g 1 &
    CUDA_VISIBLE_DEVICES=2 ./LRS -r $m -c $n -s 400 -p $(($P/$c)) -g 2 &
    CUDA_VISIBLE_DEVICES=3 ./LRS -r $m -c $n -s 600 -p $(($P/$c)) -g 3 &
    CUDA_VISIBLE_DEVICES=4 ./LRS -r $m -c $n -s 800 -p $(($P/$c)) -g 4 &
    progress 100 "Done"
elif(($c==6));
then
    CUDA_VISIBLE_DEVICES=0 ./LRS -r $m -c $n -s 0 -p $(($P/$c)) -g 0 &
    CUDA_VISIBLE_DEVICES=1 ./LRS -r $m -c $n -s 166 -p $(($P/$c)) -g 1 &
    CUDA_VISIBLE_DEVICES=2 ./LRS -r $m -c $n -s 332 -p $(($P/$c)) -g 2 &
    CUDA_VISIBLE_DEVICES=3 ./LRS -r $m -c $n -s 498 -p $(($P/$c)) -g 3 &
    CUDA_VISIBLE_DEVICES=4 ./LRS -r $m -c $n -s 664 -p $(($P/$c)) -g 4 &
    CUDA_VISIBLE_DEVICES=5 ./LRS -r $m -c $n -s 830 -p $(($P/$c + 4)) -g 5 &
    progress 100 "Done"
elif(($c==7));
then
    CUDA_VISIBLE_DEVICES=0 ./LRS -r $m -c $n -s 0 -p $(($P/$c)) -g 0 &
    CUDA_VISIBLE_DEVICES=1 ./LRS -r $m -c $n -s 142 -p $(($P/$c)) -g 1 &
    CUDA_VISIBLE_DEVICES=2 ./LRS -r $m -c $n -s 284 -p $(($P/$c)) -g 2 &
    CUDA_VISIBLE_DEVICES=3 ./LRS -r $m -c $n -s 426 -p $(($P/$c)) -g 3 &
    CUDA_VISIBLE_DEVICES=4 ./LRS -r $m -c $n -s 568 -p $(($P/$c)) -g 4 &
    CUDA_VISIBLE_DEVICES=5 ./LRS -r $m -c $n -s 710 -p $(($P/$c)) -g 5 &
    CUDA_VISIBLE_DEVICES=6 ./LRS -r $m -c $n -s 852 -p $(($P/$c + 6)) -g 6 &
    progress 100 "Done"
elif(($c==8));
then
    CUDA_VISIBLE_DEVICES=0 ./LRS -r $m -c $n -s 0 -p $(($P/$c)) -g 0 &
    CUDA_VISIBLE_DEVICES=1 ./LRS -r $m -c $n -s 125 -p $(($P/$c)) -g 1 &
    CUDA_VISIBLE_DEVICES=2 ./LRS -r $m -c $n -s 250 -p $(($P/$c)) -g 2 &
    CUDA_VISIBLE_DEVICES=3 ./LRS -r $m -c $n -s 375 -p $(($P/$c)) -g 3 &
    CUDA_VISIBLE_DEVICES=4 ./LRS -r $m -c $n -s 500 -p $(($P/$c)) -g 4 &
    CUDA_VISIBLE_DEVICES=5 ./LRS -r $m -c $n -s 625 -p $(($P/$c)) -g 5 &
    CUDA_VISIBLE_DEVICES=6 ./LRS -r $m -c $n -s 750 -p $(($P/$c)) -g 6 &
    CUDA_VISIBLE_DEVICES=7 ./LRS -r $m -c $n -s 875 -p $(($P/$c)) -g 7 &
    progress 100 "Done"
elif(($c==9));
then
    CUDA_VISIBLE_DEVICES=0 ./LRS -r $m -c $n -s 0 -p $(($P/$c)) -g 0 &
    CUDA_VISIBLE_DEVICES=1 ./LRS -r $m -c $n -s 111 -p $(($P/$c)) -g 1 &
    CUDA_VISIBLE_DEVICES=2 ./LRS -r $m -c $n -s 222 -p $(($P/$c)) -g 2 &
    CUDA_VISIBLE_DEVICES=3 ./LRS -r $m -c $n -s 333 -p $(($P/$c)) -g 3 &
    CUDA_VISIBLE_DEVICES=4 ./LRS -r $m -c $n -s 444 -p $(($P/$c)) -g 4 &
    CUDA_VISIBLE_DEVICES=5 ./LRS -r $m -c $n -s 555 -p $(($P/$c)) -g 5 &
    CUDA_VISIBLE_DEVICES=6 ./LRS -r $m -c $n -s 666 -p $(($P/$c)) -g 6 &
    CUDA_VISIBLE_DEVICES=7 ./LRS -r $m -c $n -s 777 -p $(($P/$c)) -g 7 &
    CUDA_VISIBLE_DEVICES=8 ./LRS -r $m -c $n -s 888 -p $(($P/$c + 1)) -g 8 &
    progress 100 "Done"
elif(($c==10));
then
    CUDA_VISIBLE_DEVICES=0 ./LRS -r $m -c $n -s 0 -p $(($P/$c)) -g 0 &
    CUDA_VISIBLE_DEVICES=1 ./LRS -r $m -c $n -s 100 -p $(($P/$c)) -g 1 &
    CUDA_VISIBLE_DEVICES=2 ./LRS -r $m -c $n -s 200 -p $(($P/$c)) -g 2 &
    CUDA_VISIBLE_DEVICES=3 ./LRS -r $m -c $n -s 300 -p $(($P/$c)) -g 3 &
    CUDA_VISIBLE_DEVICES=4 ./LRS -r $m -c $n -s 400 -p $(($P/$c)) -g 4 &
    CUDA_VISIBLE_DEVICES=5 ./LRS -r $m -c $n -s 500 -p $(($P/$c)) -g 5 &
    CUDA_VISIBLE_DEVICES=6 ./LRS -r $m -c $n -s 600 -p $(($P/$c)) -g 6 &
    CUDA_VISIBLE_DEVICES=7 ./LRS -r $m -c $n -s 700 -p $(($P/$c)) -g 7 &
    CUDA_VISIBLE_DEVICES=8 ./LRS -r $m -c $n -s 800 -p $(($P/$c)) -g 8 &
    CUDA_VISIBLE_DEVICES=9 ./LRS -r $m -c $n -s 900 -p $(($P/$c)) -g 9 &
    progress 100 "Done"


fi



