from re import T


print("This is a Simple Chatbot")

print("Marvis: Hello There!")
user_input = input("you: ")

for _ in range(len(user_input)):
    if user_input in ["hi", "hello"]:
        print("Marvis: Hi, how can I help you?")

    elif user_input == "who are you?":
        print("Marvis: I am your chatbot... do you remember me?")
        if user_input == "yes":
            print("Marvis: I am glad to see you remember me!")
        else:
            print("Marvis: I am sorry to hear that!")

    elif user_input == "how are you?":
        print("Marvis: I am fine, thank you for asking. What about you?")
        if user_input == "fine":
            print("Marvis: I am glad to hear that!")

    elif user_input == "what is your name?":
        print("Marvis: My name is Marvis. What is yours?")
        if user_input == "":
            print("Marvis: I am glad to hear that!")

    elif user_input in ["what is your age?", "how old are you?"]:
        print("Marvis: I am a computer program, I am not born yet. What is yours?")
        if user_input == "":
            print("Marvis: I am glad to hear that!")

    elif user_input == "what is your favorite color?":
        print("Marvis: I like blue. What is yours?")
        if user_input == "":
            print("Marvis: That is nice!")


    elif user_input == "what is your favorite food?":
        print("Marvis: I like pizza. What is yours?")
        if user_input == "":
            print("Marvis: I like that too!")

    elif user_input == "what is your favorite sport?":
        print("Marvis: I like football")
        if user_input == "football":
            print("Marvis: I like that too!")
        else:
            print("Marvis: That is a cool sport!")

    elif user_input == "Do you think Liverpool will win the Premier League?":
        print("Marvis: I think they will")
        if user_input == "Are you a Liverpool fan?":
            print("Marvis: Yes I am a Liverpool fan! What team do you support?")
            if user_input == "Liverpool":
                print("Marvis: I support Liverpool!")
            if user_input == "":
                print("Marvis: That's a good team!")

    elif user_input == "Do you think Liverpool will win the Champions League?":
        print("Marvis: I think they will")
        if user_input == "I don't think so":
            print("Marvis: I am sorry to hear that!")

    else:
        print("Marvis: I don't understand")

print(user_input)

