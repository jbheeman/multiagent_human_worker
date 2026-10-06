import json
import csv
import os
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

console = Console()

def load_pairs(filepath="evaluation_pairs.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        print("Loaded pairs from", filepath)
        return json.load(f)

def save_vote(pair_id, vote, annotator_name, output_csv="human_votes.csv"):
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Annotator', 'Pair_ID', 'Vote'])
        writer.writerow([annotator_name, pair_id, vote])

def main():
    console.clear()
    annotator_name = Prompt.ask("[bold cyan]Enter your name (Annotator)[/]")
    pairs = load_pairs()
    
    for i, pair in enumerate(pairs, 1):
        console.clear()
        console.print(f"[bold yellow]--- Evaluating Pair {i}/{len(pairs)} ---[/]")
        
        # Display Context
        console.print(Panel(
            f"[bold]Domain:[/] {pair['domain']}\n"
            f"[bold]Goal:[/] {pair['task_goal']}\n"
            f"[bold]Persona:[/] {pair['persona_profile']}",
            title="[bold cyan]Context (Read First)[/]",
            border_style="cyan"
        ))
        
        # Display Trace A
        console.print(Panel(
            pair['trace_A'], 
            title="[bold green]Trace A[/]", 
            border_style="green"
        ))
        
        # Display Trace B
        console.print(Panel(
            pair['trace_B'], 
            title="[bold blue]Trace B[/]", 
            border_style="blue"
        ))
        
        # Get Vote
        console.print("\n[bold]Which agent provided a more satisfying experience for THIS persona?[/]")
        vote = Prompt.ask(
            "Enter [bold green]A[/], [bold blue]B[/], or [bold yellow]T[/] (Tie)", 
            choices=["A", "B", "T", "a", "b", "t"]
        ).upper()
        
        # Save Result
        save_vote(pair['pair_id'], vote, annotator_name)
        console.print(f"[bold magenta]Vote '{vote}' saved![/] Moving to next...")

    console.print("\n[bold green]🎉 All done! Excellent work.[/]")

if __name__ == "__main__":
    main()