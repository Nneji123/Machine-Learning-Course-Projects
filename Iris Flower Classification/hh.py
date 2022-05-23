def fizzBuzz(n):
    i = 1
    for i in range(n):
        if i % 3 == 0 and i% 5 == 0:
            print("fizzBuzz")
        elif i%3==0:
            print("Fizz")
        elif i%5==0:
            print("Buzz")
        else:
            print(i)
x = int(input())
            
fizzBuzz(x)
