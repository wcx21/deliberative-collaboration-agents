import random


class Persona:
    def __init__(self, name, gender, nationality, age, occupation=None, recent_status=None,
                 dietary_preferences=None, food_allergies=None, cultural_preferences=None,
                 health_and_fitness_goals=None,
                 personality_and_dining_style=None, ethical_and_religious_considerations=None):
        # Assigning values or setting them to None if not provided
        # Basic personal information
        self.name = name
        self.gender = gender
        self.nationality = nationality
        self.age = age

        # Independent attributes (Influencing factors)
        self.occupation = occupation if occupation else {}
        self.recent_status = recent_status if recent_status else {}
        self.dietary_preferences = dietary_preferences if dietary_preferences else {}
        self.food_allergies = food_allergies if food_allergies else []
        self.cultural_preferences = cultural_preferences if cultural_preferences else {}
        self.health_and_fitness_goals = health_and_fitness_goals if health_and_fitness_goals else {}
        self.personality_and_dining_style = personality_and_dining_style if personality_and_dining_style else {}
        self.ethical_and_religious_considerations = ethical_and_religious_considerations if ethical_and_religious_considerations else {}

        # Derived attributes (calculated based on the provided information)
        self.values = None

    def __repr__(self):
        # Custom string representation for easier viewing
        return f"Persona(Name: {self.name}, Gender: {self.gender}, Nationality: {self.nationality}, Age: {self.age})"

    def get_full_profile(self):
        # Return a full profile of the persona in a readable format
        profile = {
            "Name": self.name,
            "Gender": self.gender,
            "Nationality": self.nationality,
            "Age": self.age,
            "Occupation": self.occupation,
            "Recent Status": self.recent_status,
            "Dietary Preferences": self.dietary_preferences,
            "Food Allergies": self.food_allergies,
            "Cultural Preferences": self.cultural_preferences,
            "Health and Fitness Goals": self.health_and_fitness_goals,
            "Personality and Dining Style": self.personality_and_dining_style,
            "Ethical and Religious Considerations": self.ethical_and_religious_considerations
        }
        return profile

    def partial_observation(self, observation_probs):
        """
        Create a new Persona instance with partial observations based on given category probabilities.
        observation_probs: Dictionary with probabilities of observing each category (range [0, 1]).
        """

        def observe(prob):
            return random.random() < prob

        # Observing the basic information
        observed_name = self.name if observe(observation_probs["Name"]) else None
        observed_gender = self.gender if observe(observation_probs["Gender"]) else None
        observed_nationality = self.nationality if observe(observation_probs["Nationality"]) else None
        observed_age = self.age if observe(observation_probs["Age"]) else None

        # Observing categories
        observed_occupation = self.occupation if observe(observation_probs["Occupation"]) else None
        observed_recent_status = self.recent_status if observe(observation_probs["Recent Status"]) else None
        observed_dietary_preferences = self.dietary_preferences if observe(
            observation_probs["Dietary Preferences"]) else None
        observed_food_allergies = self.food_allergies if observe(observation_probs["Food Allergies"]) else None
        observed_cultural_preferences = self.cultural_preferences if observe(
            observation_probs["Cultural Preferences"]) else None
        observed_health_and_fitness_goals = self.health_and_fitness_goals if observe(
            observation_probs["Health and Fitness Goals"]) else None
        observed_personality_and_dining_style = self.personality_and_dining_style if observe(
            observation_probs["Personality and Dining Style"]) else None
        observed_ethical_and_religious_considerations = self.ethical_and_religious_considerations if observe(
            observation_probs["Ethical and Religious Considerations"]) else None

        # Return a new instance with the partially observed attributes
        return Persona(
            observed_name, observed_gender, observed_nationality, observed_age,
            observed_occupation, observed_recent_status, observed_dietary_preferences,
            observed_food_allergies, observed_cultural_preferences, observed_health_and_fitness_goals,
            observed_personality_and_dining_style, observed_ethical_and_religious_considerations
        )

def create_persona_from_dict(persona_dict):
    '''
    example:
    {
        "Name": "Alice",
        "Gender": "Female",
        "Nationality": "American",
        "Age": 30,
        "Occupation": {
            "Job Title": "Software Engineer",
            "Industry": "Technology",
            "Work Schedule": "9am - 5pm"
        },
        "Recent Status": {
            "Current Focus": "Project deadline approaching",
            "Recent Dietary Focus": "Looking for healthy, comforting meals to recharge",
            "Stress Level": "High"
        },
        "Dietary Preferences": {
            "Vegetarian": true,
            "Vegan": false,
            "Gluten-free": true,
            "Dairy-free": false,
            "Low-carb": false
        },
        "Food Allergies": ["Nuts", "Dairy"],
        "Cultural Preferences": {
            "Cuisine Type": "Italian",
            "Spice Level Preference": "Mild",
            "Traditional Dishes": true
        },
        "Health and Fitness Goals": {
            "Calorie-conscious": true,
            "Protein-rich": false,
            "Low-sodium": false,
            "Low-sugar": false,
            "Weight-loss focus": false,
            "Muscle-building focus": false
        },
        "Personality and Dining Style": {
            "Adventurous Eater": false,
            "Comfort Food Lover": true,
            "Indulgent Food Preferences": true,
            "Minimalist Dining": false,
            "Social Eater": true,
            "Snack-friendly": true
        },
        "Ethical and Religious Considerations": {
            "Halal": false,
            "Kosher": false,
            "Buddhist": false,
            "Ethical Eating Preferences": "Organic"
        }
    }
    :param persona_dict:
    :return:
    '''
    name = persona_dict.get("Name")
    gender = persona_dict.get("Gender")
    nationality = persona_dict.get("Nationality")
    age = persona_dict.get("Age")
    occupation = persona_dict.get("Occupation")
    recent_status = persona_dict.get("Recent Status")
    dietary_preferences = persona_dict.get("Dietary Preferences")
    food_allergies = persona_dict.get("Food Allergies")
    cultural_preferences = persona_dict.get("Cultural Preferences")
    health_and_fitness_goals = persona_dict.get("Health and Fitness Goals")
    personality_and_dining_style = persona_dict.get("Personality and Dining Style")
    ethical_and_religious_considerations = persona_dict.get("Ethical and Religious Considerations")

    persona = Persona(
        name, gender, nationality, age,
        occupation, recent_status, dietary_preferences,
        food_allergies, cultural_preferences, health_and_fitness_goals,
        personality_and_dining_style, ethical_and_religious_considerations
    )
    return persona


if __name__ == '__main__':
    # Example of how to use the Persona class and partial observation
    observation_probs = {
        "Name": 1.0,
        "Gender": 1.0,
        "Nationality": 1.0,
        "Age": 1.0,
        "Occupation": 0.6,
        "Recent Status": 0.6,
        "Dietary Preferences": 0.7,
        "Food Allergies": 0.8,
        "Cultural Preferences": 0.6,
        "Health and Fitness Goals": 0.5,
        "Personality and Dining Style": 0.7,
        "Ethical and Religious Considerations": 0.6
    }

    # Create the original persona
    david = Persona(
        "David", "Male", "Canadian", 38,
        {
            "Job Title": "Architect",
            "Industry": "Construction",
            "Work Schedule": "9am - 5pm"
        },
        {
            "Current Focus": "Designing new office building plans",
            "Recent Dietary Focus": "Eating balanced meals for better productivity",
            "Stress Level": "Medium"
        },
        {
            "Vegetarian": False,
            "Vegan": False,
            "Gluten-free": True,
            "Dairy-free": False,
            "Low-carb": False
        },
        ["Eggs"],
        {
            "Cuisine Type": "French",
            "Spice Level Preference": "Mild",
            "Traditional Dishes": True
        },
        {
            "Calorie-conscious": False,
            "Protein-rich": True,
            "Low-sodium": True,
            "Low-sugar": False,
            "Weight-loss focus": False,
            "Muscle-building focus": True
        },
        {
            "Adventurous Eater": False,
            "Comfort Food Lover": True,
            "Indulgent Food Preferences": True,
            "Minimalist Dining": False,
            "Social Eater": True,
            "Snack-friendly": True
        },
        {
            "Halal": False,
            "Kosher": False,
            "Buddhist": False,
            "Ethical Eating Preferences": "Cruelty-free"
        }
    )

    # Create a partially observed instance
    partial_david = david.partial_observation(observation_probs)

    # You can access partial_david's attributes here
    print(partial_david)  # Will print the string representation of the persona
    print(partial_david.get_full_profile())  # Will print the full profile of the persona



