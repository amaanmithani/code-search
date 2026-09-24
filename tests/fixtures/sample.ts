import { thing } from "./thing";

/** Shape of a user record. */
export interface User {
  id: number;
}

export class UserStore extends Base {
  private users: User[] = [];

  addUser(user: User): void {
    this.users.push(user);
  }
}

export function formatName(first: string, last: string): string {
  return `${first} ${last}`;
}

const toUpper = (s: string) => s.toUpperCase();
