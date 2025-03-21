# ECE-226-MeZO-Survey

Q0. Given a memory/accuracy contraint and the type of task I'm working with, what's the method/model that can achieve the best accuracy/memory? (mostly PEFT)


Q1. What are the pros and cons of each category of methods? For example, hybrid gives the best accuracy when fine-tuning small models in language comprehension tasks.	


Q2. Does model size matter?


Q3. Which category of method is the most unstable (high vairance across tasks/models)?


How to interpolate [method A (with category A') 
				    fine-tuning model B (with model size B')
					on dataset C (with task type C')]?

                    
	Fix k of A, A', B, B', C, C' and take the avg

    
	Q1: Fix A' B' C' (cuz we're asking for a specific model size and task type)

    
	Q2: 