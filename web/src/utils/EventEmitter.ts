/**
 * Preprost EventEmitter za uporabo v brskalniku
 * Nadomešča Node.js EventEmitter, ki ni na voljo v brskalniku
 */
export class EventEmitter {
  private events: Record<string, Array<(...args: any[]) => void>> = {};

  /**
   * Registrira poslušalca za dogodek
   * @param event Ime dogodka
   * @param listener Funkcija, ki se pokliče ob dogodku
   */
  public on(event: string, listener: (...args: any[]) => void): this {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(listener);
    return this;
  }

  /**
   * Odstrani poslušalca za dogodek
   * @param event Ime dogodka
   * @param listener Funkcija, ki se odstrani
   */
  public removeListener(event: string, listener: (...args: any[]) => void): this {
    if (!this.events[event]) {
      return this;
    }
    
    const index = this.events[event].indexOf(listener);
    if (index !== -1) {
      this.events[event].splice(index, 1);
    }
    return this;
  }

  /**
   * Odstrani vse poslušalce za dogodek
   * @param event Ime dogodka
   */
  public removeAllListeners(event?: string): this {
    if (event) {
      delete this.events[event];
    } else {
      this.events = {};
    }
    return this;
  }

  /**
   * Sproži dogodek
   * @param event Ime dogodka
   * @param args Argumenti, ki se posredujejo poslušalcem
   */
  public emit(event: string, ...args: any[]): boolean {
    if (!this.events[event]) {
      return false;
    }
    
    this.events[event].forEach(listener => {
      listener(...args);
    });
    
    return true;
  }

  /**
   * Vrne število poslušalcev za dogodek
   * @param event Ime dogodka
   */
  public listenerCount(event: string): number {
    return this.events[event]?.length || 0;
  }

  /**
   * Vrne vse poslušalce za dogodek
   * @param event Ime dogodka
   */
  public listeners(event: string): Array<(...args: any[]) => void> {
    return this.events[event] || [];
  }

  /**
   * Registrira poslušalca za dogodek, ki se pokliče samo enkrat
   * @param event Ime dogodka
   * @param listener Funkcija, ki se pokliče ob dogodku
   */
  public once(event: string, listener: (...args: any[]) => void): this {
    const onceWrapper = (...args: any[]) => {
      this.removeListener(event, onceWrapper);
      listener(...args);
    };
    
    return this.on(event, onceWrapper);
  }
}
