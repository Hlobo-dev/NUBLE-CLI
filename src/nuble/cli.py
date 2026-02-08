import sys
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from nuble.manager import Manager
from nuble import console

# Maximum conversation messages before auto-compact (prevents token explosion)
MAX_CONVERSATION_MESSAGES = 40

# Check if services are available for status display
try:
    from nuble.services import get_services
    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False

# Check if Lambda client is available
try:
    from nuble.lambda_client import get_lambda_client, NUBLE_API_BASE
    LAMBDA_AVAILABLE = True
except ImportError:
    LAMBDA_AVAILABLE = False
    NUBLE_API_BASE = None

def display_application_banner():
    banner_text = """
███╗   ██╗██╗   ██╗██████╗ ██╗     ███████╗
████╗  ██║██║   ██║██╔══██╗██║     ██╔════╝
██╔██╗ ██║██║   ██║██████╔╝██║     █████╗  
██║╚██╗██║██║   ██║██╔══██╗██║     ██╔══╝  
██║ ╚████║╚██████╔╝██████╔╝███████╗███████╗
╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝
"""
    
    # Create gradient effect similar to Gemini CLI
    lines = banner_text.strip().split('\n')
    gradient_colors = ['bright_blue', 'blue', 'cyan', 'bright_cyan', 'magenta', 'bright_magenta']
    
    styled_lines = []
    for i, line in enumerate(lines):
        color = gradient_colors[i % len(gradient_colors)]
        styled_lines.append(Text(line, style=f"bold {color}"))
    
    # Combine all lines
    full_banner = Text()
    full_banner.append("\n\n")
    for line in styled_lines:
        full_banner.append(line)
        full_banner.append('\n')
    
    # Add subtitle with gradient
    subtitle = Text("Institutional-Grade AI Investment Research", style="bold bright_magenta")
    full_banner.append('\n')
    full_banner.append(subtitle)
    
    # Print banner without border, left-aligned like Gemini CLI
    console.print(full_banner)


def show_system_status():
    """Display system status and capabilities."""
    if not SERVICES_AVAILABLE:
        console.print("[yellow]Services not fully initialized[/yellow]")
        return
    
    try:
        import asyncio
        services = get_services()
        
        # Initialize services if not already done
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, services.initialize())
                future.result(timeout=10)
        except RuntimeError:
            asyncio.run(services.initialize())
        
        status = services.get_status()
        
        # Build status table
        table = Table(title="🔧 NUBLE System Status", border_style="bright_cyan")
        table.add_column("Component", style="white")
        table.add_column("Status", style="white")
        table.add_column("Details", style="dim")
        
        for service_type, svc_status in status.items():
            status_icon = "✅" if svc_status.available else "❌"
            status_text = f"{status_icon} {'Available' if svc_status.available else 'Unavailable'}"
            table.add_row(svc_status.name, status_text, str(svc_status.details)[:50])
        
        console.print(table)
        console.print()
        
        # Show Lambda status
        if LAMBDA_AVAILABLE:
            try:
                client = get_lambda_client()
                console.print("[bold bright_cyan]🌐 Lambda Decision Engine:[/bold bright_cyan]")
                console.print(f"  [green]Connected[/green] → {NUBLE_API_BASE}")
                console.print("  [dim]Real-time: StockNews (24), CryptoNews (17), Polygon[/dim]")
                console.print()
            except:
                console.print("[yellow]Lambda: Not connected[/yellow]\n")
        
        # Show LuxAlgo status
        console.print("[bold bright_cyan]📊 LuxAlgo Premium Signals:[/bold bright_cyan]")
        console.print("  [dim]Multi-timeframe (W/D/4H) via TradingView → DynamoDB → 34% Decision Engine weight[/dim]")
        console.print("  [dim]Use /luxalgo SYMBOL to check signal status[/dim]")
        console.print()
        
        # Show TENK status
        try:
            from nuble.agents.fundamental_analyst import FundamentalAnalystAgent
            fa = FundamentalAnalystAgent()
            tenk_ok = fa._init_tenk()
            if tenk_ok:
                filings = fa._tenk_db.list_filings()
                console.print(f"[bold bright_cyan]📄 TENK SEC Filing RAG:[/bold bright_cyan]")
                console.print(f"  [green]Connected[/green] — {len(filings)} filing(s) loaded")
                if filings:
                    tickers = set(f['ticker'] for f in filings)
                    console.print(f"  [dim]Tickers: {', '.join(sorted(tickers))}[/dim]")
                console.print(f"  [dim]Use /tenk SYMBOL to search filings[/dim]")
            else:
                console.print("[bold bright_cyan]📄 TENK SEC Filing RAG:[/bold bright_cyan]")
                console.print("  [yellow]Not initialized[/yellow]")
        except Exception:
            pass
        console.print()
        
        # Show capabilities
        console.print("[bold bright_cyan]Available Commands:[/bold bright_cyan]")
        console.print("  [white]AAPL[/white]          - Quick quote")
        console.print("  [white]predict TSLA[/white]  - ML prediction")
        console.print("  [white]RSI for AMD[/white]   - Technical analysis")
        console.print("  [white]NVDA 10-K[/white]     - SEC filings")
        console.print("  [white]/lambda BTC[/white]   - Direct Lambda API test")
        console.print("  [white]/luxalgo TSLA[/white] - LuxAlgo premium signals")
        console.print("  [white]/tenk AAPL[/white]    - SEC filing RAG search")
        console.print("  [white]/status[/white]       - System status")
        console.print("  [white]/help[/white]         - Full command list")
        console.print()
        
    except Exception as e:
        console.print(f"[red]Could not get system status: {e}[/red]")


def handle_quick_command(cmd: str, manager: Manager, messages: list = None) -> bool:
    """
    Handle quick commands that start with /
    Returns True if command was handled.
    """
    cmd_lower = cmd.lower().strip()
    
    if cmd_lower == '/status':
        show_system_status()
        return True
    
    if cmd_lower in ('/quit', '/exit', '/q'):
        console.print("\n[bright_cyan]Goodbye![/bright_cyan]")
        sys.exit(0)
    
    if cmd_lower == '/version':
        from nuble import __version__
        console.print(f"[bright_cyan]NUBLE[/bright_cyan] v{__version__} - APEX PREDATOR Edition")
        console.print("[dim]Architecture: APEX Dual-Brain Fusion (9 agents + DecisionEngine + ML)[/dim]")
        console.print("[dim]ML Models: MLP, LSTM, Transformer, N-BEATS, Ensemble (46M+ params)[/dim]")
        console.print("[dim]Data Sources: Polygon.io, LuxAlgo Premium, SEC EDGAR, StockNews (24), CryptoNews (17)[/dim]")
        console.print("[dim]Filing RAG: TENK SEC 10-K/10-Q (DuckDB + sentence-transformers)[/dim]")
        if LAMBDA_AVAILABLE:
            console.print(f"[dim]Lambda API: [green]Connected[/green] → {NUBLE_API_BASE}[/dim]")
        else:
            console.print("[dim]Lambda API: [yellow]Not connected[/yellow][/dim]")
        return True
    
    if cmd_lower == '/clear':
        console.clear()
        if messages is not None:
            messages.clear()
        display_application_banner()
        console.print("[green]Screen and conversation history cleared.[/green]\n")
        return True
    
    if cmd_lower == '/help':
        console.print("\n[bold bright_cyan]NUBLE Commands:[/bold bright_cyan]\n")
        console.print("  [white]/status[/white]          — System component status")
        console.print("  [white]/lambda SYMBOL[/white]   — Direct Lambda Decision Engine test")
        console.print("  [white]/luxalgo SYMBOL[/white]  — LuxAlgo premium signal status")
        console.print("  [white]/tenk SYMBOL[/white]     — SEC filing RAG search")
        console.print("  [white]/version[/white]         — Version info")
        console.print("  [white]/clear[/white]           — Clear screen + history")
        console.print("  [white]/quit[/white]            — Exit")
        console.print("\n[bold bright_cyan]Query Examples:[/bold bright_cyan]\n")
        console.print("  [white]AAPL[/white]             — Quick quote")
        console.print("  [white]predict TSLA[/white]     — ML prediction")
        console.print("  [white]RSI for AMD[/white]      — Technical analysis")
        console.print("  [white]Should I buy NVDA?[/white] — APEX deep analysis (9 agents)")
        console.print("  [white]Why is BTC down?[/white] — Crypto analysis")
        console.print()
        return True

    # LuxAlgo signal check command
    if cmd_lower.startswith('/luxalgo'):
        parts = cmd_lower.split()
        if len(parts) < 2:
            console.print("[yellow]Usage: /luxalgo SYMBOL[/yellow]")
            console.print("[dim]Example: /luxalgo TSLA[/dim]")
            return True
        
        symbol = parts[1].upper()
        console.print(f"[dim]Fetching LuxAlgo signals for {symbol}...[/dim]")
        try:
            if not LAMBDA_AVAILABLE:
                console.print("[yellow]Lambda client not available — LuxAlgo signals come via Lambda[/yellow]")
                return True
            
            client = get_lambda_client()
            analysis = client.get_analysis(symbol)
            
            console.print(f"\n[bold bright_magenta]📊 LuxAlgo Premium Signals: {symbol}[/bold bright_magenta]")
            console.print(f"[bold white]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold white]")
            
            if analysis.luxalgo_valid_count > 0 or analysis.luxalgo_aligned:
                aligned_text = "[green]✅ ALL TIMEFRAMES ALIGNED[/green]" if analysis.luxalgo_aligned else "[yellow]⚠️  Mixed signals[/yellow]"
                console.print(f"  Alignment: {aligned_text}")
                console.print(f"  Direction: [white]{analysis.luxalgo_direction or 'N/A'}[/white]")
                console.print(f"  Weekly (1W): [white]{analysis.luxalgo_weekly_action}[/white]  [dim]— highest conviction[/dim]")
                console.print(f"  Daily  (1D): [white]{analysis.luxalgo_daily_action}[/white]")
                console.print(f"  4-Hour (4H): [white]{analysis.luxalgo_h4_action}[/white]  [dim]— most responsive[/dim]")
                console.print(f"  Score: [white]{analysis.luxalgo_score:+.3f}[/white]  [dim](34% weight in Decision Engine)[/dim]")
                console.print(f"  Valid Signals: [white]{analysis.luxalgo_valid_count}/3[/white] timeframes")
                if analysis.luxalgo_aligned:
                    console.print(f"\n  [bold green]🔥 HIGH CONVICTION — All timeframes agree on {analysis.luxalgo_direction}[/bold green]")
            else:
                console.print("  [dim]No LuxAlgo signals available for this symbol.[/dim]")
                console.print("  [dim]Check TradingView webhooks → DynamoDB pipeline.[/dim]")
            
            console.print()
        except Exception as e:
            console.print(f"[red]LuxAlgo error: {e}[/red]")
        return True
    
    # TENK SEC Filing RAG command
    if cmd_lower.startswith('/tenk'):
        parts = cmd_lower.split()
        if len(parts) < 2:
            console.print("[yellow]Usage: /tenk SYMBOL[/yellow]")
            console.print("[dim]Example: /tenk AAPL[/dim]")
            return True
        
        symbol = parts[1].upper()
        console.print(f"[dim]Searching SEC filings for {symbol}...[/dim]")
        try:
            from nuble.agents.fundamental_analyst import FundamentalAnalystAgent
            agent = FundamentalAnalystAgent()
            insights = agent._get_tenk_filing_insights(symbol)
            
            console.print(f"\n[bold bright_cyan]📄 TENK SEC Filing RAG: {symbol}[/bold bright_cyan]")
            console.print(f"[bold white]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold white]")
            
            if insights.get('available'):
                filings = insights.get('filings_loaded', [])
                console.print(f"  Filings Loaded: [white]{len(filings)}[/white]")
                for f in filings[:5]:
                    q_str = f" Q{f['quarter']}" if f.get('quarter') else ""
                    console.print(f"    • {f['form']} {f['year']}{q_str} ({f['chunks']} chunks)")
                
                for topic in ['risk_factors', 'revenue_breakdown', 'management_outlook', 'competitive_position']:
                    results = insights.get(topic, [])
                    if results:
                        topic_display = topic.replace('_', ' ').title()
                        console.print(f"\n  [bold]{topic_display}:[/bold]")
                        for r in results[:2]:
                            text = r['text'][:200] + '...' if len(r['text']) > 200 else r['text']
                            console.print(f"    [dim]({r['form']} {r['year']}, score={r['score']:.3f})[/dim]")
                            console.print(f"    {text}")
            else:
                reason = insights.get('reason', 'Unknown')
                console.print(f"  [yellow]{reason}[/yellow]")
                if insights.get('hint'):
                    console.print(f"  [dim]{insights['hint']}[/dim]")
            
            console.print()
        except Exception as e:
            console.print(f"[red]TENK error: {e}[/red]")
        return True

    # Lambda status/test command
    if cmd_lower.startswith('/lambda'):
        if not LAMBDA_AVAILABLE:
            console.print("[red]Lambda client not available[/red]")
            return True
        
        parts = cmd_lower.split()
        if len(parts) > 1:
            symbol = parts[1].upper()
            console.print(f"[dim]Fetching real-time data for {symbol} from Lambda...[/dim]")
            try:
                client = get_lambda_client()
                analysis = client.get_analysis(symbol)
                
                console.print(f"\n[bold bright_cyan]NUBLE Decision Engine: {symbol}[/bold bright_cyan]")
                console.print(f"[bold]Action: {analysis.action}[/bold] | Score: {analysis.score}/100 | Confidence: {analysis.confidence}")
                
                if analysis.current_price > 0:
                    change_color = "green" if analysis.change_percent >= 0 else "red"
                    console.print(f"Price: ${analysis.current_price:.2f} [{change_color}]{analysis.change_percent:+.2f}%[/{change_color}]")
                
                if analysis.rsi > 0:
                    console.print(f"RSI: {analysis.rsi:.1f} | MACD: {analysis.macd:.4f} | VIX: {analysis.vix:.1f}")
                
                if analysis.stocknews_summary:
                    console.print(f"\n[bold]StockNews:[/bold] {analysis.stocknews_summary[:200]}...")
                
                if analysis.cryptonews_summary:
                    console.print(f"\n[bold]CryptoNews:[/bold] {analysis.cryptonews_summary[:200]}...")
                
                # Show LuxAlgo signals if available
                if analysis.luxalgo_valid_count > 0:
                    aligned_text = "✅ ALIGNED" if analysis.luxalgo_aligned else "⚠️ Mixed"
                    console.print(f"\n[bold bright_magenta]LuxAlgo:[/bold bright_magenta] {aligned_text} | W:{analysis.luxalgo_weekly_action} D:{analysis.luxalgo_daily_action} 4H:{analysis.luxalgo_h4_action} | Score: {analysis.luxalgo_score:+.3f}")
                
                console.print()
            except Exception as e:
                console.print(f"[red]Lambda error: {e}[/red]")
        else:
            # Just show Lambda status
            try:
                client = get_lambda_client()
                health = client.health_check()
                console.print(f"[green]Lambda API: Connected[/green]")
                console.print(f"[dim]Endpoint: {NUBLE_API_BASE}[/dim]")
                console.print(f"[dim]Status: {health.get('status', 'unknown')}[/dim]")
            except Exception as e:
                console.print(f"[red]Lambda API error: {e}[/red]")
        return True

    return False  # Not a quick command

def interactive_shell():
    display_application_banner()
    
    # Show Lambda connection status
    if LAMBDA_AVAILABLE:
        try:
            client = get_lambda_client()
            console.print("\n[green]✓ Lambda Decision Engine Connected[/green]")
            console.print("[dim]  Real-time data: StockNews, CryptoNews, Polygon.io[/dim]")
        except:
            console.print("\n[yellow]⚠ Lambda Decision Engine not responding[/yellow]")
    
    # Show APEX status
    try:
        from nuble.agents.orchestrator import OrchestratorAgent
        console.print("[green]✓ APEX Dual-Brain Fusion Active[/green]")
        console.print("[dim]  9 specialist agents • DecisionEngine • ML Predictor[/dim]")
    except ImportError:
        console.print("[yellow]⚠ APEX tier not available (single-brain mode)[/yellow]")
    
    # Tips section for user guidance
    console.print("\n[dim white]Tips for getting started:[/dim white]")
    console.print("[white]1. Quick: [/white][bright_cyan]AAPL[/bright_cyan][white] → instant quote[/white]")
    console.print("[white]2. Crypto: [/white][bright_cyan]Why is BTC down?[/bright_cyan][white] → real-time crypto analysis[/white]")
    console.print("[white]3. ML: [/white][bright_cyan]predict TSLA[/bright_cyan][white] → AI prediction[/white]")
    console.print("[white]4. Technical: [/white][bright_cyan]RSI for AMD[/bright_cyan][white] → indicators[/white]")
    console.print("[white]5. Research: [/white][bright_cyan]Should I buy NVDA?[/bright_cyan][white] → APEX deep analysis (9 agents)[/white]")
    console.print("[white]6. Lambda: [/white][bright_cyan]/lambda BTC[/bright_cyan][white] → direct API test[/white]")
    console.print("[white]7. LuxAlgo: [/white][bright_cyan]/luxalgo TSLA[/bright_cyan][white] → premium multi-TF signals[/white]")
    console.print("[white]8. SEC Filings: [/white][bright_cyan]/tenk AAPL[/bright_cyan][white] → SEC filing RAG search[/white]")
    console.print("[white]9. Commands: [/white][bright_cyan]/help[/bright_cyan][white], [/white][bright_cyan]/status[/bright_cyan][white], [/white][bright_cyan]/version[/bright_cyan][white], [/white][bright_cyan]/clear[/bright_cyan]\n")
    
    # Use free agent by default
    selected_agent = Manager()
    
    print("\nType your queries below. Press Ctrl+C to exit.\n")
    messages = []
    try:
        while True:
            # Display prompt for user input
            console.print("[bright_cyan]> [/bright_cyan]", end="")
            try:
                user_input_text = input()
            except EOFError:
                print("\n\nGoodbye!")
                sys.exit(0)
            
            if not user_input_text.strip():
                console.print("[yellow]Please enter a query.[/yellow]\n")
                continue
            
            # Handle quick commands first (don't add to conversation)
            if user_input_text.startswith('/'):
                if handle_quick_command(user_input_text, selected_agent, messages):
                    continue
            
            # Auto-compact conversation if it's getting too long
            if len(messages) > MAX_CONVERSATION_MESSAGES:
                console.print("[dim]🔄 Auto-compacting conversation to save tokens...[/dim]")
                try:
                    selected_agent.agent.compact(messages)
                    console.print(f"[dim]   Compacted to {len(messages)} messages[/dim]")
                except Exception:
                    # Fallback: just keep last 10 messages
                    messages[:] = messages[-10:]
            
            # Process normal query — process_prompt handles message appending
            messages.append({"role": "user", "content": user_input_text})
            selected_agent.process_prompt(user_input_text, messages)
                
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)

def _cleanup():
    """Clean up resources on exit. Suppress asyncio unclosed session warnings."""
    import warnings
    import logging
    # Suppress the asyncio "Unclosed client session" warnings at exit.
    # These come from aiohttp sessions in orchestrator agent threads that ran
    # via asyncio.run() — they're cleaned up by GC but asyncio logs a warning
    # before that happens. Safe to suppress since the process is exiting.
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)
    warnings.filterwarnings('ignore', message='Unclosed client session')
    warnings.filterwarnings('ignore', message='Unclosed connector')


def main():
    # Register cleanup
    import atexit
    atexit.register(_cleanup)
    interactive_shell()

if __name__ == '__main__':
    main()