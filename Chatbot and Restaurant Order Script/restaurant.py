print("This is a Simple Food Delivery Script Made with Python ")
def steps():
    # Opening the app
    open_app = input("Do you want to use the app? Type y if yes.\n")
    if open_app == "y":
        print("Opening app...")
    else:
        return "Exit"
    
    # Entering Email and Password
    print("Input Your Username/Email Address and Password")
    user_login_email = input("Email address: ")
    user_login_password = input("Password: ")
    
    
    order_food = input("Do you want to order food? type y if yes.\n")
    if order_food == "y":
        print("Ordering food")
    else:
        return "Okay no problem"
    type_of_food = input("What kind of food do you want?\n")
    price = len(type_of_food)
    print(f"You ordered {type_of_food}")
    if price>=5:
        print("This will cost you:$ ", price*100)
    else:
        print("This will cost you:$ ", price*50)
    
    
    available_restaurants = ["Uber Eats", "The Place "]
    print(f"These are the restaurants available {available_restaurants}")
    select_restaurant = input("What Restaurant Do you want to order from: ")
    if select_restaurant in available_restaurants:
        print(f"Ordering from {select_restaurant}")
    else:
        print("The restaurant you entered ", select_restaurant, " is not available")
    
    location = input("Do you want your food delivered? Type y for Yes\n")
    if location == "y":
        delivery_locations = ["Ajah", "VGC", "Lekki", "Ikorodu"]
        print(f"These are our Delivery locations\n {delivery_locations} ")
        user_location=input("Choose a location:\n")
        if user_location in delivery_locations:
            print("Your food is going to be delivered to this location: ", user_location)
        else:
            print("This location is not available")
            
    list_of_payment = ["Bank Transfer", "Credit Card", "Pay on Delivery"]
    print("These are the payment methods that are available", list_of_payment)
    user_payment = input("Select your payment method:\n")
    if user_payment in list_of_payment:
        print(f"You have chosen the {user_payment} method of payment\n")
        print("Your order has now been placed!\n")
        print("This is your order summary:\n")
        print(f"Username: {user_login_email}\n")
        print(f"What you ordered: {type_of_food}\n")
        print(f"Price of food: {len(type_of_food)}\n")
        print(f"Restaurant Selected: {select_restaurant}")
        print(f"Delivery Location: {user_location}\n")
        print(f"Payment Method: {user_payment}")
    else:
        print("This payment method is either unavailable or not offered by us.\nPlease Choose from the available payment methods")
        print("Your order is invalid. Please try again!")
        
    
    
steps()