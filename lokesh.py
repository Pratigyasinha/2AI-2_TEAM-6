class giMember7:
    """
    giMember7 Class
    
    This class represents a group member and stores
    information related to member details.

    Attributes
    ----------
    name : str
        Name of the member
    age : int
        Age of the member
    role : str
        Role assigned in the group

    Methods
    -------
    display_info()
        Prints member information
    """

    def __init__(self, name, age, role):
        """
        Constructs all necessary attributes for giMember7

        Parameters
        ----------
        name : str
            Name of the member
        age : int
            Age of the member
        role : str
            Role in group
        """
        self.name = name
        self.age = age
        self.role = role

    def display_info(self):
        """
        Displays the member information
        """
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")
        print(f"Role: {self.role}")
        member = giMember7("Rahul", 22, "Developer")
member.display_info()
