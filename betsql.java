// table student
// id,name,dept

// table sub
// id,stud_id,sub,percentage


// *crud operation
// *get student List with student name with subject name
// *get student list which who has subject is Math
// *get student List with subject which has percentage is greater than 60 %

class student {
    private int id;
    private String name;
    private String dept;

    public student(int id, String name, String dept) {
        this.id = id;
        this.name = name;
        this.dept = dept;
    }

    public void display() {
        System.out.println("ID: " + id + ", Name: " + name + ", Dept: " + dept);
    }   
}

class sub {
    private int id;
    private int stud_id;
    private String sub;
    private double percentage;

    public sub(int id, int stud_id, String sub, double percentage) {
        this.id = id;
        this.stud_id = stud_id;
        this.sub = sub;
        this.percentage = percentage;
    }
    public void display() {
        System.out.println("ID: " + id + ", Student ID: " + stud_id + ", Subject: " + sub + ", Percentage: " + percentage);
    }
}

public class bet {
    public static void main(String[] args) {
        student s1 = new student(1, "Deven", "CS");
        student s2 = new student(2, "Yash", "IT");
        sub sub1 = new sub(1, 1, "Math", 75.0);
        sub sub2 = new sub(2, 1, "Science", 85.0);
        sub sub3 = new sub(3, 2, "Math", 65.0);
        
        s1.display();
        s2.display();
        sub1.display();
        sub2.display();
        sub3.display();

        
    }
}
