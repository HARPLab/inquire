class Demonstration:
    def __repr__(self):
        return f"{self.__class__.__name__}"

    @staticmethod
    def present_to_user(query):
        print("Placeholder for posing a Demonstration query to the user")

class Correction:
    def __repr__(self):
        return f"{self.__class__.__name__}"

    @staticmethod
    def present_to_user(query):
        print("Placeholder for posing a Correction query to the user")

class Preference:
    def __repr__(self):
        return f"{self.__class__.__name__}"

    @staticmethod
    def present_to_user(query):
        print("Placeholder for posing a Preference query to the user")

class BinaryFeedback:
    def __repr__(self):
        return f"{self.__class__.__name__}"

    @staticmethod
    def present_to_user(query):
        print("Placeholder for posing a Binary Feedback query to the user")
